# This file is part of pysimgrid, a Python interface to the SimGrid library.
#
# Copyright 2015-2016 Alexey Nazarenko and contributors
#
# This library is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# along with this library.  If not, see <http://www.gnu.org/licenses/>.
#

import abc
import itertools
import logging
import os
import time

from enum import Enum

from .. import six
from .. import csimdag
from .. import cplatform


class TaskExecutionMode(Enum):
  """
  Execution mode defines how tasks are executed on a host.

  - SEQUENTIAL (default):
    task are executed strictly one by one, in order specified by the scheduler.

  - PARALLEL:
    tasks can execute in parallel, host speed is fairly shared between concurrent tasks.
  """
  SEQUENTIAL = 1
  PARALLEL = 2


class DataTransferMode(Enum):
  """
  Data transfer strategy defines when and in what order data transfers, corresponding to edges in a workflow DAG,
  are scheduled during the workflow execution. For each data transfer, the source task is called producer and
  the destination task is called consumer. Applicable for SEQUENTIAL execution mode only.

  - EAGER (default):
    Data transfer is scheduled when the data is ready, i.e. the producer is completed,
    and the destination node is known, i.e. the consumer is scheduled.

  - LAZY:
    Data transfer is scheduled when the destination node is ready to execute the consumer task.

  - PREFETCH:
    Data transfer is scheduled when the destination node is ready to execute a task
    immediately preceding the consumer task.

  - QUEUE:
    Data transfers on each destination node are scheduled sequentially in the order of planned execution
    of consumer tasks on this node.

  - QUEUE_ECT:
    Data transfers on each destination node are scheduled sequentially in the order of expected completion time
    of producer tasks, breaking the ties with the order of planned execution of consumer tasks.

  - PARENTS:
    Data transfer is scheduled when all parents of the consumer task are completed.

  - LAZY_PARENTS:
    Combination of LAZY and PARENTS strategies.
  """
  EAGER = 1
  LAZY = 2
  PREFETCH = 3
  QUEUE = 4
  QUEUE_ECT = 5
  PARENTS = 6
  LAZY_PARENTS = 7
  EAGER_CACHING = 8


class Scheduler(six.with_metaclass(abc.ABCMeta)):
  """
  Base class for all scheduling algorithms.

  Defines scheduler public interface and provides (very few) useful methods for
  actual schedulers:

    *self._log* - Logger object (see logging module documentation)

    *self._check_done()* - raise an exception if there are any unfinished tasks
  """
  BOUNDARY_TASKS = ["root", "end"]
  MASTER_HOST_NAME = "master"

  def __init__(self, simulation):
    """
    Initialize scheduler instance.

    Args:
      simulation: a :class:`pysimgrid.simdag.Simulation` object
    """
    self._simulation = simulation
    self._log = logging.getLogger(type(self).__name__)

    # Task execution and data transfer modes are configured via environment variables.
    if "PYSIMGRID_TASK_EXECUTION" in os.environ:
      self._task_exec_mode = TaskExecutionMode[os.environ["PYSIMGRID_TASK_EXECUTION"]]
    else:
      self._task_exec_mode = TaskExecutionMode.SEQUENTIAL
    if "PYSIMGRID_DATA_TRANSFER" in os.environ:
      self._data_transfer_mode = DataTransferMode[os.environ["PYSIMGRID_DATA_TRANSFER"]]
    else:
      self._data_transfer_mode = DataTransferMode.EAGER

    algo = type(self).__name__
    if self._data_transfer_mode == DataTransferMode.QUEUE_ECT:
      if algo not in ['HEFT', 'Lookahead']:
        raise Exception('%s does not support %s mode' % (algo, self._data_transfer_mode))

  @abc.abstractmethod
  def run(self):
    """
    Descibes the simulation process.
    Single call to this method should perform the full
    simulation, scheduling all the tasks and calling the
    simulation API as necessary.

    Note:
      Not intended to be overriden in the concrete algorithms.
    """
    raise NotImplementedError()

  @abc.abstractproperty
  def scheduler_time(self):
    """
    Wall clock time spent scheduling.
    """
    raise NotImplementedError()

  @abc.abstractproperty
  def total_time(self):
    """
    Wall clock time spent scheduling and simulating.
    """
    raise NotImplementedError()

  @property
  def expected_makespan(self):
    """
    Algorithm's makespan prediction. Can return None if algorithms didn't/cannot provide it.
    """
    return None

  def _check_done(self):
    """
    Check that all tasks are completed after the simulation.

    Useful to transform SimGrid's unhappy logs into actual detectable error.
    """
    unfinished = self._simulation.all_tasks.by_prop("state", csimdag.TASK_STATE_DONE, True)
    if any(unfinished):
      raise Exception("some tasks are not finished by the end of simulation: {}".format([t.name for t in unfinished]))

  @classmethod
  def is_boundary_task(cls, task):
    return (task.name in cls.BOUNDARY_TASKS) and task.amount == 0


class StaticScheduler(Scheduler):
  """
  Base class for static scheduling algorithms.

  Provides some non-trivial functionality, ensuring that tasks scheduled on the same host
  do not execute concurrently.
  """
  def __init__(self, simulation):
    super(StaticScheduler, self).__init__(simulation)
    self.__scheduler_time = -1.
    self.__total_time = -1.
    self.__expected_makespan = None

  def run(self):
    start_time = time.time()

    schedule = self.get_schedule(self._simulation)

    self.__scheduler_time = time.time() - start_time
    self._log.debug("Scheduling time: %f", self.__scheduler_time)
    if not isinstance(schedule, (dict, tuple)):
      raise Exception("'get_schedule' must return a dictionary or a tuple")
    if isinstance(schedule, tuple):
      if len(schedule) != 2 or not isinstance(schedule[0], dict) or not isinstance(schedule[1], float):
        raise Exception("'get_schedule' returned tuple should have format (<schedule>, <expected_makespan>)")
      schedule, self.__expected_makespan = schedule
      self._log.debug("Expected makespan: %f", self.__expected_makespan)
    for host, task_list in schedule.items():
      if not (isinstance(host, cplatform.Host) and isinstance(task_list, list)):
        raise Exception("'get_schedule' must return a dictionary Host:List_of_tasks")

    unscheduled = self._simulation.tasks[csimdag.TASK_STATE_NOT_SCHEDULED, csimdag.TASK_STATE_SCHEDULABLE]
    if set(itertools.chain.from_iterable(schedule.values())) != set(self._simulation.tasks):
      raise Exception("some tasks are left unscheduled by static algorithm: {}".format([t.name for t in unscheduled]))
    if len(unscheduled) != len(self._simulation.tasks):
      raise Exception("static scheduler should not directly schedule tasks")

    task_to_host = {}
    for host, tasks in schedule.items():
      for task in tasks:
        task_to_host[task] = host

    if self._data_transfer_mode in (DataTransferMode.EAGER, DataTransferMode.PARENTS,DataTransferMode.PREFETCH,
                                    DataTransferMode.QUEUE, DataTransferMode.QUEUE_ECT):
      comm_tasks = [task for task in self._simulation._tasks if task.kind == csimdag.TaskKind.TASK_KIND_COMM_E2E]
      for comm_task in comm_tasks:
        producer = comm_task.parents[0]
        consumer = comm_task.children[0]
        dummy_task = self._simulation.add_task('__DUMMY__TRANSFER__TASK__{}_{}'.format(producer.name, consumer.name), 1)
        self._simulation.remove_dependency(comm_task, consumer)
        self._simulation.add_dependency(comm_task, dummy_task)
        dummy_transfer_task = self._simulation.add_transfer('{}_e2e'.format(dummy_task.name), 1)
        self._simulation.add_dependency(dummy_task, dummy_transfer_task)
        self._simulation.add_dependency(dummy_transfer_task, consumer)
    elif self._data_transfer_mode == DataTransferMode.EAGER_CACHING:
      comm_tasks = [task for task in self._simulation._tasks if task.kind == csimdag.TaskKind.TASK_KIND_COMM_E2E]
      for comm_task in comm_tasks:
        dummy_task = self._simulation.add_task('__DUMMY__TRANSFER__TASK__{}'.format(comm_task), 1)
        self._simulation.add_dependency(comm_task, dummy_task)
        for consumer in comm_task.children:
          self._simulation.remove_dependency(comm_task, consumer)
          dummy_transfer_task = self._simulation.add_transfer('{}_e2e'.format(dummy_task.name), 1)
          self._simulation.add_dependency(dummy_task, dummy_transfer_task)
          self._simulation.add_dependency(dummy_transfer_task, consumer)

    if self._data_transfer_mode in [DataTransferMode.QUEUE, DataTransferMode.QUEUE_ECT]:
      data_transfers = []
      for host, tasks in schedule.items():
        for pos, task in enumerate(tasks):
          if self._data_transfer_mode == DataTransferMode.QUEUE:
            # build a list of host inbound data transfers
            for transfer_task in task.parents:
              data_transfers.append((transfer_task, pos))
          elif self._data_transfer_mode == DataTransferMode.QUEUE_ECT:
            # build a list of host inbound data transfers
            for transfer_task in task.parents:
              producer = transfer_task.parents[0].parents[0]
              data_transfers.append((transfer_task, (producer.data["ect"], pos)))

      # form a queue from host inbound data transfers
      data_transfers.sort(key=lambda t: t[1])
      data_transfers_per_host = {
        host: [t for t, _ in data_transfers if task_to_host[t.children[0]] == host]
        for host in schedule.keys()
      }
      for transfers in data_transfers_per_host.values():
        prev_comm = None
        for comm in transfers:
          if prev_comm is not None:
            self._simulation.add_dependency(prev_comm, comm)
          prev_comm = comm

    if self._data_transfer_mode == DataTransferMode.PREFETCH:
      for host, tasks in schedule.items():
        for prev, nxt in zip(tasks[:-1], tasks[1:]):
          for prev_comm in prev.parents:
            for nxt_comm in nxt.parents:
              self._simulation.add_dependency(prev_comm, nxt_comm)

    hosts_status = {h: h.cores for h in self._simulation.hosts}
    for t in self._simulation.tasks:
      t.watch(csimdag.TASK_STATE_DONE)
    changed = self._simulation.tasks.by_func(lambda t: False)
    while True:
      for t in changed.by_prop("kind", csimdag.TASK_KIND_COMM_E2E, negate=True)[csimdag.TASK_STATE_DONE]:
        if t.name.startswith('__DUMMY__TRANSFER__TASK__'):
          continue  # special dummy tasks do not utilize CPU
        for h in t.hosts:
          hosts_status[h] += 1
        if self._data_transfer_mode in (DataTransferMode.EAGER, DataTransferMode.PARENTS,
                                        DataTransferMode.PREFETCH, DataTransferMode.EAGER_CACHING,
                                        DataTransferMode.QUEUE, DataTransferMode.QUEUE_ECT):
          self.check_and_schedule_parent_transfers(t, task_to_host)
      for host, tasks in schedule.items():
        scheduled_tasks = set()
        for task in tasks:
          if not hosts_status[host] and not self._task_exec_mode == TaskExecutionMode.PARALLEL:
            break
          if self._data_transfer_mode == DataTransferMode.LAZY_PARENTS:
            if not all(p.parents[0].state == csimdag.TASK_STATE_DONE for p in task.parents):
              continue
          scheduled_tasks.add(task)
          task.schedule(host)
          hosts_status[host] -= 1
        schedule[host] = [task for task in tasks if task not in scheduled_tasks]
      changed = self._simulation.simulate()
      if not changed:
        break

    self._simulation.simulate()
    self._check_done()
    self.__total_time = time.time() - start_time

  def check_and_schedule_parent_transfers(self, task, task_to_host):
    for e2e in task.children:
      dummy_task = e2e.children[0]
      if self._data_transfer_mode == DataTransferMode.PARENTS:
        consumer = dummy_task.children[0].children[0]
        consumer_transfers = [x.parents[0] for x in consumer.parents]
        consumer_parents = [x.parents[0].parents[0] for x in consumer_transfers]
        if all(x.state == csimdag.TASK_STATE_DONE for x in consumer_parents):
          for transfer_task in consumer_transfers:
            consumer_task = transfer_task.children[0].children[0]
            host = task_to_host[consumer_task]
            if transfer_task.state < csimdag.TASK_STATE_SCHEDULED:
              transfer_task.schedule(host)
      else:
        dummy_task = e2e.children[0]
        consumer_tasks = [child.children[0] for child in dummy_task.children]
        noncached_tasks = [x for x in consumer_tasks if task.name not in x.data.get('uses_cache_for', [])]
        assert len(noncached_tasks) == 1
        consumer_task = noncached_tasks[0]
        host = task_to_host[consumer_task]
        dummy_task.schedule(host)

  @abc.abstractmethod
  def get_schedule(self, simulation):
    """
    Abstract method that need to be overriden in scheduler implementation.

    Args:
      simulation: a :class:`pysimgrid.simdag.Simulation` object

    Returns:

      Expected to return a schedule as dict {host -> [list_of_tasks...]}.
      Optionally, can also return a predicted makespan. Then return type is a tuple (schedule, predicted_makespan_in_seconds).
    """
    raise NotImplementedError()

  @property
  def scheduler_time(self):
    return self.__scheduler_time

  @property
  def total_time(self):
    return self.__total_time

  @property
  def expected_makespan(self):
    return self.__expected_makespan


class DynamicScheduler(Scheduler):
  """
  Base class for dynamic scheduling algorithms.
  """
  def __init__(self, simulation):
    super(DynamicScheduler, self).__init__(simulation)
    self.__scheduler_time = -1.
    self.__total_time = -1.

  def run(self):
    start_time = time.time()
    self.prepare(self._simulation)
    for t in self._simulation.tasks:
      t.watch(csimdag.TASK_STATE_DONE)
    scheduler_time = time.time()
    # a bit ugly kludge - cannot just pass an empty list there, needs to be a _TaskList
    self.schedule(self._simulation, self._simulation.all_tasks.by_func(lambda t: False))
    self.__scheduler_time = time.time() - scheduler_time
    changed = self._simulation.simulate()
    while changed:
      scheduler_time = time.time()
      self.schedule(self._simulation, changed)
      self.__scheduler_time += time.time() - scheduler_time
      changed = self._simulation.simulate()
    self._check_done()
    self.__total_time = time.time() - start_time

  @abc.abstractmethod
  def prepare(self, simulation):
    """
    Abstract method that need to be overriden in scheduler implementation.

    Executed once before the simulation. Can be used to setup initial state for tasks and hosts.

    Args:
      simulation: a :class:`pysimgrid.simdag.Simulation` object
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def schedule(self, simulation, changed):
    """
    Abstract method that need to be overriden in scheduler implementation.

    Args:
      simulation: a :class:`pysimgrid.simdag.Simulation` object
      changed: a list of changed tasks
    """
    raise NotImplementedError()

  @property
  def scheduler_time(self):
    return self.__scheduler_time

  @property
  def total_time(self):
    return self.__total_time
