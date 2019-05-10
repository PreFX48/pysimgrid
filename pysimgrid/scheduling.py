import bisect
import heapq
import networkx
import numpy

from . import csimdag
from . import cplatform
from .cscheduling import MinSelector, try_schedule_boundary_task_object as try_schedule_boundary_task



class PlatformModel(object):
  """
  Platform linear model used for most static scheduling approaches.

  Disregards network topology.
  """

  def __init__(self, simulation):
    hosts = simulation.hosts
    speed = numpy.zeros(len(hosts))
    bandwidth = numpy.zeros((len(hosts), len(hosts)))
    latency = numpy.zeros((len(hosts), len(hosts)))
    for i, src in enumerate(hosts):
      speed[i] = src.speed
      for j in range(i+1, len(hosts)):
        dst = simulation.hosts[j]
        bandwidth[i,j] = bandwidth[j,i] = cplatform.route_bandwidth(src, dst)
        latency[i,j] = latency[j,i] = cplatform.route_latency(src, dst)

    self._speed = speed
    self._bandwidth = bandwidth
    self._latency = latency
    self._mean_speed = speed.mean()
    self._mean_bandwidth = bandwidth.mean() * (bandwidth.size / (bandwidth.size - len(hosts)))
    self._mean_latency = latency.mean() * (latency.size / (bandwidth.size - len(hosts)))
    self._host_map = {host: idx for (idx, host) in enumerate(hosts)}

  @property
  def host_count(self):
    """
    Get platform host count.
    """
    return len(self._speed)

  @property
  def speed(self):
    """
    Get hosts speed as a vector.

    Refer to to host_map property or host_idx function to convert host instances to indices.
    """
    return self._speed

  @property
  def bandwidth(self):
    """
    Get platform connection bandwidths as a matrix.

    Note:
      For i==j bandwidth is 0
    """
    return self._bandwidth

  @property
  def latency(self):
    """
    Get platform connection latencies as a matrix.

    Note:
      For i==j latency is 0
    """
    return self._latency

  @property
  def mean_speed(self):
    """
    Get mean host speed in a platform.
    """
    return self._mean_speed

  @property
  def mean_bandwidth(self):
    """
    Get mean connection bandwidth in a platform.
    """
    return self._mean_bandwidth

  @property
  def mean_latency(self):
    """
    Get mean connection latency in a platform.
    """
    return self._mean_latency

  @property
  def host_map(self):
    """
    Get {Host: idx} mapping.
    """
    return self._host_map

  def eet(self, task, host):
    """
    Calculate task eet on a given host.
    """
    return task.amount / host.speed

  def parent_data_ready_time(self, host, parent, edge_dict, state):
    """
    Calculate data ready time for a single parent.

    Args:
      host: host on which a new (current) task will be executed
      parent: parent task
      edge_dict: edge properties dict (for now the only important property is "weight")
      state: current schedule state

    Return:
      earliest start time considering only a given parent
    """
    task_states = state.task_states
    dst_idx = self._host_map[host]
    src_idx = self._host_map[task_states[parent]["host"]]
    if src_idx == dst_idx:
      return state.task_states[parent]["ect"]
    return task_states[parent]["ect"] + edge_dict["weight"] / self._bandwidth[src_idx, dst_idx] + self._latency[src_idx, dst_idx]

  def est(self, host, parents, state):
    result = 0.
    task_states = state._task_states
    dst_idx = self._host_map[host]
    bw = self._bandwidth
    lat = self._latency
    for parent, edge_dict in parents.items():
      task_state = task_states[parent]
      src_idx = self._host_map[task_state["host"]]
      if src_idx == dst_idx:
        parent_time = task_state["ect"]
      else:
        comm_amount = edge_dict["weight"]
        # extract ect first to ensure it has fixed type
        # otherwise + operator will trigger nasty python lookup
        parent_time = task_state["ect"]
        parent_time += comm_amount / bw[src_idx, dst_idx] + lat[src_idx, dst_idx]
      if parent_time > result:
        result = parent_time
    return result

  def enhanced_est(self, host, parents, state, use_cache):  # TODO: object -> bool?
    """
    Calculate an earliest start time for a given task while also taking into account network contention
    """
    task_states = state._task_states

    transfers = [
      {
        'name': edge['name'],
        'parent': parent.name,
        'start_time': task_states[parent]['ect'],
        'src': task_states[parent]['host'],
        'dst': host,
        'amount': edge['weight'],
      }
      for parent, edge in parents.items()
    ]
    transfer_finish_time, cached_tasks, transfer_finishes = state.get_transfer_time(
      transfers,
      absolute=True,
      use_cache=use_cache,
    )
    return transfer_finish_time, cached_tasks, transfer_finishes

  def max_ect(self, tasks, state):
    """
    Get max ECT (estimated completion time) for given tasks.
    """
    result = 0.
    task_states = state._task_states

    for task in tasks:
      task_state = task_states[task]
      task_ect = task_state["ect"]
      if task_ect > result:
        result = task_ect
    return result

  def enhanced_max_comm_time(self, host, tasks, state):
    """
    Calculate an earliest start time for a given task while also taking into account network contention
    """
    task_states = state._task_states
    transfers = [
      {
        'name': edge['name'],
        'start_time': task_states[parent]['ect'],
        'src': task_states[parent]['host'],
        'dst': host,
        'amount': edge['weight'],
      }
      for parent, edge in tasks.items()
    ]
    transfer_duration = state.get_transfer_time(transfers, absolute=False, use_cache=False)  # TODO: вызываемая функция изменилась
    return transfer_duration

  def host_idx(self, host):
    return self._host_map[host]


class SchedulerState(object):
  """
  Stores the current scheduler state.

  See properties description for details.
  """

  def __init__(self, simulation=None, task_states=None, timetable=None, links=None,
               transfer_tasks=None, cached_transfers=None, hosts=None, tasks=None):
    if simulation:
      if task_states or timetable:
        raise Exception("simulation is provided, initial state is not expected")
      self._task_states = {task: {"ect": numpy.nan, "host": None} for task in simulation.tasks}
      self._timetable = {host: [] for host in simulation.hosts}
      self._links = set()
      for h1 in simulation.hosts:
        for h2 in simulation.hosts:
          for link in cplatform.route(h1, h2):
            self._links.add(link)
      self._transfer_tasks = {task: None for task in simulation.connections}
      self._transfer_task_by_name = {task.name: task for task in self._transfer_tasks}
      self._cached_transfers = {}
      self._hosts = simulation.hosts
      self._tasks = [t for t in simulation.all_tasks]
    else:
      if (
        not task_states or not timetable or not links or not transfer_tasks
        or cached_transfers is None or not hosts or not tasks):
        raise Exception("initial state must be provided")
      self._task_states = task_states
      self._timetable = timetable
      self._links = links
      self._transfer_tasks = transfer_tasks
      self._transfer_task_by_name = {task.name: task for task in self._transfer_tasks}
      self._cached_transfers = cached_transfers
      self._hosts = hosts
      self._tasks = tasks

  def get_transfer_time(self, new_tasks, absolute, use_cache=False):
    transfer_finishes = {}
    cached_tasks = set()
    for task in new_tasks:
      if task['src'] == task['dst']:
        transfer_finishes[task['name']] = task['start_time']
        continue
      cache_time = self._cached_transfers.get(task['parent'], {}).get(task['dst'].name)
      if use_cache and cache_time is not None:
        transfer_finishes[task['name']] = max(task['start_time'], cache_time[0])
        cached_tasks.add(task['name'])
    new_tasks = [task for task in new_tasks if task['src'] != task['dst'] and task['name'] not in cached_tasks]


    if not new_tasks:
      if absolute:
        return max(transfer_finishes.values()), cached_tasks, transfer_finishes
      else:
        return 0.0, cached_tasks, transfer_finishes

    transfer_tasks = sorted(
      [(t, info) for (t, info) in self._transfer_tasks.items() if info is not None and info[1] != info[2]],
      key=lambda x: x[1][0]
    )
    task_to_links = {task['name']: cplatform.route(task['src'], task['dst']) for task in new_tasks}
    for task, task_info in transfer_tasks:
      if task_info is not None and task_info[1] != task_info[2]:
        task_to_links[task.name] = cplatform.route(task_info[1], task_info[2])
    link_usage = {name: 0 for name in self._links}
    link_bandwidth = {name: cplatform.link_bandwidth(name) for name in self._links}

    cur_time = 0
    transfer_tasks_idx = 0
    to_transfer = {}
    unfinished_tasks = {task['name'] for task in new_tasks}
    unscheduled_tasks = {task['name'] for task in new_tasks}
    if not absolute:
      results = {}
      task_to_start_time = {task['name']: task['start_time'] for task in new_tasks}

    while True:
      selector = MinSelector()
      if transfer_tasks_idx < len(transfer_tasks):
        task, task_info = transfer_tasks[transfer_tasks_idx]
        selector.update((task_info[0],), (task.name, 'scheduled_task', task.amount))
      for task, amount_left in to_transfer.items():
        effective_speed = min(link_bandwidth[link] / (link_usage[link]+1) for link in task_to_links[task])
        eta = amount_left / effective_speed
        selector.update((cur_time + eta,), (task, 'finished_task', 0))
      for task in new_tasks:
        if task['name'] in unscheduled_tasks:
          selector.update((task['start_time'],), (task['name'], 'scheduled_task', task['amount']))
      event_time = selector.key[0]
      affected_task, event_type, task_amount = selector.value

      if event_type == 'scheduled_task':
        if affected_task in unscheduled_tasks:
          unscheduled_tasks.remove(affected_task)
        else:
          transfer_tasks_idx += 1
      if affected_task in unfinished_tasks and event_type == 'finished_task':
        transfer_finishes[affected_task] = event_time
        unfinished_tasks.remove(affected_task)
        if not absolute:
          results[affected_task] = event_time - task_to_start_time[affected_task]
      if not unfinished_tasks:
        if absolute:
          return max(transfer_finishes.values()), cached_tasks, transfer_finishes
        else:
          return max(results), cached_tasks, transfer_finishes

      for task in to_transfer:
        effective_speed = min(link_bandwidth[link] / (link_usage[link]+1) for link in task_to_links[task])
        to_transfer[task] -= (event_time - cur_time) * effective_speed
      if event_type == 'scheduled_task':
        for link in task_to_links[affected_task]:
          link_usage[link] += 1
        to_transfer[affected_task] = task_amount
      elif event_type == 'finished_task':
        for link in task_to_links[affected_task]:
          link_usage[link] -= 1
        del to_transfer[affected_task]
      cur_time = event_time

  def update_schedule_for_transfers(self, new_tasks):
    # SIMULATE THE WORLD
    for task in new_tasks:
      self._transfer_tasks[task['task']] = (task['start_time'], task['src'], task['dst'])
    import os
    if os.environ.get('IMPROVE_SIMULATION', '0') == '0':
      return

    task_to_links = {
      task: cplatform.route(task_info[1], task_info[2])
      for (task, task_info) in self._transfer_tasks.items()
      if task_info is not None and task_info[1] != task_info[2]
    }
    comp_timetable = {}
    for host, timesheet in self._timetable.items():
      for task, start_time, _ in timesheet:
        comp_timetable[task] = (host, start_time)
    pending_dependencies = {task: len(task.parents) for task in self._tasks}
    awaiting_children = {task: task.children for task in self._tasks}
    link_usage = {name: 0 for name in self._links}
    link_bandwidth = {name: cplatform.link_bandwidth(name) for name in self._links}

    root = [task for task in self._tasks if task.name == 'root']
    assert len(root) == 1
    root = root[0]

    comp_events = []
    heapq.heappush(comp_events, (self._task_states[root]['ect'], 0, root))
    to_transfer = {}

    cur_time = None  # will be filled at the first iteration
    while comp_events or to_transfer:
      if comp_events:
        comp_time, comp_type, comp_task = comp_events[0]
      else:
        comp_time = None
      if to_transfer:
        selector = MinSelector()
        for task, amount_left in to_transfer.items():
          effective_speed = min(link_bandwidth[link] / (link_usage[link] + 1) for link in task_to_links[task])
          eta = amount_left / effective_speed
          assert cur_time is not None
          selector.update((cur_time + eta,), (task,))
        comm_time = selector.key[0]
        comm_task = selector.value[0]
      else:
        comm_time = None

      if comm_time is not None and (comp_time is None or comm_time <= comp_time):
        for task in to_transfer:
          effective_speed = min(link_bandwidth[link] / (link_usage[link] + 1) for link in task_to_links[task])
          to_transfer[task] -= (comm_time - cur_time) * effective_speed
        for link in task_to_links[comm_task]:
          link_usage[link] -= 1
        del to_transfer[comm_task]
        for child in awaiting_children[comm_task]:
          pending_dependencies[child] -= 1
          if pending_dependencies[child] == 0:
            expected_host, expected_time = comp_timetable[child]
            heapq.heappush(comp_events, (max(expected_time, comm_time), 1, child))
        cur_time = comm_time
      else:
        heapq.heappop(comp_events)
        assert comp_type in (0, 1)
        if comp_type == 0:
          for task in to_transfer:
            effective_speed = min(link_bandwidth[link] / (link_usage[link] + 1) for link in task_to_links[task])
            to_transfer[task] -= (comp_time - cur_time) * effective_speed
          for child in comp_task.children:
            if not self._transfer_tasks.get(child):
              continue  # this task have not been scheduled yet
            old_info = self._transfer_tasks[child]
            assert self._transfer_tasks.get(child)
            print(old_info[1].name)
            if old_info[1].name != 'master':
              import ipdb; ipdb.set_trace(context=9)
            self._transfer_tasks[child] = (comp_time, old_info[1], old_info[2])
            to_transfer[child] = child.amount
            for link in task_to_links[child]:
              link_usage[link] += 1
          cur_time = comp_time
        elif comp_type == 1:
          self._shift_schedule(self._task_states[comp_task]['host'], comp_task, comp_time)

  def _shift_schedule(self, host, task, start_time):
    tasks = [
      ((task, start_time, start_time+self._task_states[task]['eet']) if (t[0] == task and t[1] < start_time) else t)
      for t in self._timetable[host]
    ]
    self._timetable[host] = []
    for task, start_time, finish_time in tasks:
      pos, start, finish = timesheet_insertion(self._timetable[host], host.cores, start_time, finish_time-start_time)
      self.update(task, host, pos, start, finish)


  def __shift_schedule_2(self, host, task, start_time):
    raise NotImplementedError('It can be finished, but for now we just reuse timesheet_insertion method')

    timesheet = self._timetable[host]
    beginnings = [(task[1], 1, task[0]) for task in timesheet]
    finishes = [(task[2], 0, task[0]) for task in timesheet]
    events = sorted(finishes + beginnings)
    available_cores = host.cores

    task_pos = [i for (i, t) in enumerate(timesheet) if t[0] == task][0]
    if start_time <= timesheet[task_pos][1]:
      return
    timesheet[task_pos] = (task, start_time, start_time+self._task_states[task]['eet'])

    finishes = []
    idx = 0
    cur_time = 0
    while finishes or idx < len(timesheet):
      next_start = None
      next_finish = None
      if finishes:
        next_finish = finishes[0][1]
      if available_cores and idx < len(timesheet):
        next_start = timesheet[idx][1]
      if next_start is not None and (next_finish is None or next_start < next_finish):
        next_task = timesheet[idx][0]
        self._task_states[next_task]['est'] = next_start
        self._task_states[next_task]['ect'] = next_start + self._task_states[next_task]['eet']
        heapq.heappush(finishes, (next_start+next_finish))
        available_cores -= 1
        idx += 1
        cur_time = next_start
      else:
        heapq.heappop(finishes)
        next_task = finishes[0][0]
        cur_time = next_finish


  def copy(self):
    """
    Return a deep (enough) copy of a state.

    Timesheet tuples aren't actually copied, but they shouldn't be modified anyway.

    Note:
      Exists purely for optimization. copy.deepcopy is just abysmally slow.
    """
    # manual copy of initial state
    #   copy.deepcopy is slow as hell
    task_states = {task: dict(state) for (task, state) in self._task_states.items()}
    timetable = {host: list(timesheet) for (host, timesheet) in self._timetable.items()}
    links = {link for link in self._links}
    transfer_tasks = {task: value for (task, value) in self._transfer_tasks.items()}
    cached_transfers = {k1: {k2: v2 for (k2, v2) in v1.items()} for (k1, v1) in self._cached_transfers.items()}
    hosts = self._hosts  # is not modifiable
    tasks = [t for t in self._tasks]
    return SchedulerState(
      task_states=task_states,
      timetable=timetable,
      links=links,
      transfer_tasks=transfer_tasks,
      cached_transfers=cached_transfers,
      hosts=hosts,
      tasks=tasks,
    )

  @property
  def task_states(self):
    """
    Get current task states as a dict.

    Layout: a dict {Task: {"ect": float, "host": Host}}
    """
    return self._task_states

  @property
  def transfer_tasks(self):
    return self._transfer_tasks

  @property
  def transfer_task_by_name(self):
    return self._transfer_task_by_name

  @property
  def cached_transfers(self):
    return self._cached_transfers

  @property
  def timetable(self):
    """
    Get a timesheets dict.

    Layout: a dict {Host: [(Task, start, finish)...]}
    """
    return self._timetable

  @property
  def schedule(self):
    """
    Get a schedule from a current timetable.

    Layout: a dict {Host: [Task...]}
    """
    return {host: [task for (task, _, _) in timesheet] for (host, timesheet) in self._timetable.items()}

  @property
  def max_time(self):
    """
    Get a finish time of a last scheduled task in a state.

    Returns NaN if no tasks are scheduled.
    """
    finish_times = [s["ect"] for s in self._task_states.values() if numpy.isfinite(s["ect"])]
    return numpy.nan if not finish_times else max(finish_times)

  def update(self, task, host, pos, start, finish):
    """
    Update timetable for a given host.

    Note:
      Doesn't perform any validation for now, can produce overlapping timesheets if used carelessly.
      Checks can be costly.


    Args:
      task: task to schedule on a host
      host: host considered
      pos: insertion position
      start: task start time
      finish: task finish time
    """
    # update task state
    task_state = self._task_states[task]
    timesheet = self._timetable[host]
    task_state["est"] = start
    task_state["eet"] = finish - start
    task_state["ect"] = finish
    task_state["host"] = host
    # update timesheet
    timesheet.insert(pos, (task, start, finish))

  def remove_transfer(self, task):
    if task.kind != csimdag.TASK_KIND_COMM_E2E:
      raise ValueError("Task's type should be TASK_KIND_COMM_E2E")
    del self.transfer_tasks[task]
    del self.transfer_task_by_name[task.name]
    self._tasks = [t for t in self._tasks if t != task]


def is_master_host(host):
  MASTER_NAME = "master"
  return host.name == MASTER_NAME


def heft_order(nxgraph, platform_model):
  """
  Order task according to HEFT ranku.

  Args:
    nxgraph: full task graph as networkx.DiGraph
    platform_model: cscheduling.PlatformModel instance

  Returns:
    a list of tasks in a HEFT order
  """
  mean_speed = platform_model.mean_speed
  mean_bandwidth = platform_model.mean_bandwidth
  mean_latency = platform_model.mean_latency
  task_ranku = {}
  for idx, task in enumerate(list(reversed(list(networkx.topological_sort(nxgraph))))):
    ecomt_and_rank = [
      task_ranku[child] + (edge["weight"] / mean_bandwidth + mean_latency)
      for child, edge in nxgraph[task].items()
    ] or [0]
    task_ranku[task] = task.amount / mean_speed + max(ecomt_and_rank) + 1
  # use node name as an additional sort condition to deal with zero-weight tasks (e.g. root)
  return sorted(nxgraph.nodes(), key=lambda node: (task_ranku[node], node.name), reverse=True)


def enhanced_heft_schedule(simulation, nxgraph, platform_model, state, ordered_tasks, data_transfer_mode, is_simulated):
  """
  Builds a HEFT schedule, additionally taking into account network contention during file transfers.
  """
  for task in ordered_tasks:
    if try_schedule_boundary_task(task, nxgraph, platform_model, state):
      continue
    current_min = MinSelector()
    for host, timesheet in state.timetable.items():
      if host.name == 'master':
        continue
      if data_transfer_mode == 'EAGER_CACHING':
        parents = dict(nxgraph.pred[task])
        est, cached_tasks, transfer_finishes = platform_model.enhanced_est(host, parents, state, True)
        eet = platform_model.eet(task, host)
      else: # classic HEFT, i.e. EAGER data transfers
        est, cached_tasks, transfer_finishes = platform_model.enhanced_est(host, dict(nxgraph.pred[task]), state, False)
        eet = platform_model.eet(task, host)
      pos, start, finish = timesheet_insertion(timesheet, host.cores, est, eet)
      current_min.update((finish, host.speed, host.name), (host, pos, start, finish, cached_tasks, transfer_finishes))
    host, pos, start, finish, cached_tasks, transfer_finishes = current_min.value
    state.update(task, host, pos, start, finish)
    new_transfers = []
    if data_transfer_mode == 'EAGER_CACHING':
      for parent, edge in nxgraph.pred[task].items():
        transfer_task = state._transfer_task_by_name[edge['name']]
        state.cached_transfers.setdefault(transfer_task.parents[0].name, {})
        cached_hosts = state.cached_transfers[transfer_task.parents[0].name]
        if edge['name'] in cached_tasks:
          _, old_transfer = state.cached_transfers[parent.name][host.name]
          assert len(transfer_task.parents) == 1
          if not is_simulated:
            state.remove_transfer(transfer_task)
            simulation.remove_dependency(transfer_task.parents[0], transfer_task)
            simulation.remove_dependency(transfer_task, task)
            simulation.remove_task(transfer_task)
            simulation.add_dependency(old_transfer, task)
            if task.data is None:
              task.data = {}
            task.data.setdefault('uses_cache_for', [])
            task.data['uses_cache_for'].append(parent.name)
        else:
          if not cached_hosts.get(host.name) or cached_hosts[host.name][0] > transfer_finishes[transfer_task.name]:
            cached_hosts[host.name] = (transfer_finishes[transfer_task.name], transfer_task)
          new_transfers.append({
            'task': transfer_task,
            'start_time': state._task_states[parent]['ect'],
            'src': state._task_states[parent]['host'],
            'dst': host,
          })
    else:
      for parent, edge in nxgraph.pred[task].items():
        new_transfers.append({
          'task': state._transfer_task_by_name[edge['name']],
          'start_time': state._task_states[parent]['ect'],
          'src': state._task_states[parent]['host'],
          'dst': host,
        })
    state.update_schedule_for_transfers(new_transfers)
  return state

def timesheet_insertion(timesheet, cores, est, eet):
  """
  Evaluate a earliest possible insertion into a given timesheet.

  Args:
    timesheet: list of scheduled tasks in a form (Task, start, finish)
    est: new task earliest start time
    eet: new task execution time

  Returns:
    a tuple (insert_index, start, finish)
  """
  # implementation may look a bit ugly, but it's for performance reasons
  start_time = min([task[2] for task in timesheet[-cores:]]) if timesheet else 0
  beginnings = [(task[1], 1) for task in timesheet]
  finishes = [(task[2], 0) for task in timesheet]
  events = sorted(finishes + beginnings)
  available_cores = cores
  slot_start = 0.

  for event in events:
    if event[1] == 0:
      available_cores += 1
      if available_cores == 1:
        slot_start = event[0]
    else:
      available_cores -= 1
      if available_cores == 0:
        slot_end = event[0]
        slot = slot_end - max(slot_start, est)
        if slot > eet:
          start_time = slot_start
          break

  start_time = max(start_time, est)
  insert_index = bisect.bisect_right(beginnings, (start_time, 1))
  return (insert_index, start_time, (start_time + eet))
