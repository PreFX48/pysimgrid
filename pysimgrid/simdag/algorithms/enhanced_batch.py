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

import logging
import numpy as np
import time

from .. import scheduler
from ... import csimdag
from ... import cplatform


class EnhancedBatchScheduler(scheduler.DynamicScheduler):
    """
    Batch-mode heuristic base implementation.

    Schedules all currently schedulable tasks to a best host by ECT in a single batch.

    The order in a batch is determined by a heuristic:

    * MinMin prioritizes the tasks with minimum ECT on a best host

    * MaxMin prioritizes the tasks with maximum ECT on a best host

    * Sufferage prioritizes tasks with maximum difference between ECT on 2 best hosts
    """

    def run(self):
        start_time = time.time()
        self.prepare(self._simulation)
        for t in self._simulation.tasks:
            t.watch(csimdag.TASK_STATE_DONE)
        # ------ new piece
        for t in self._simulation.connections:
            t.watch(csimdag.TASK_STATE_RUNNING)
            t.watch(csimdag.TASK_STATE_DONE)
        # ------ end of the new piece
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

    def prepare(self, simulation):
        for h in simulation.hosts:
            h.data = {
                "est": {}
            }
        master_hosts = simulation.hosts.by_prop("name", self.MASTER_HOST_NAME)
        self._master_host = master_hosts[0] if master_hosts else None

        self.host_tasks = {}
        self._exec_hosts = simulation.hosts.by_prop("name", self.MASTER_HOST_NAME, True)
        for host in self._exec_hosts:
            self.host_tasks[host.name] = []

        if self._master_host:
            for task in simulation.tasks.by_func(lambda t: t.name in self.BOUNDARY_TASKS):
                task.schedule(self._master_host)
        self._started_tasks = set()
        self._estimate_cache = {}
        self._bandwidth = {}
        for h1 in simulation.hosts:
            for h2 in simulation.hosts:
                if h1 != h2:
                    self._bandwidth[(h1, h2)] = cplatform.route_bandwidth(h1, h2)
        links = set()
        for h1 in simulation.hosts:
            for h2 in simulation.hosts:
                for link in cplatform.route(h1, h2):
                    links.add(link)
        self._link_transmissions = {link: 0 for link in links}
        self._cached_tasks = {}
        self._first_iteration = True

    def schedule(self, simulation, changed):
        has_comp_tasks = False
        for task in changed:
            if task.kind == csimdag.TASK_KIND_COMM_E2E:
                if len(task.hosts) == 1:
                    continue
                if task.state == csimdag.TASK_STATE_DONE:
                    self._cached_tasks[(task.parents[0], task.hosts[1])] = task
                route = cplatform.route(task.hosts[0], task.hosts[1])
                for link in route:
                    if task.state == csimdag.TASK_STATE_RUNNING:
                        self._link_transmissions[link] += 1
                    elif task.state == csimdag.TASK_STATE_DONE:
                        self._link_transmissions[link] -= 1
            elif task.kind == csimdag.TASK_KIND_COMP_SEQ:
                has_comp_tasks = True
        if not has_comp_tasks and not self._first_iteration:
            return
        self._first_iteration = False

        clock = simulation.clock

        available_cores = {host: host.cores for host in self._exec_hosts}
        tasks_to_remove = []

        for task in simulation.tasks[
            csimdag.TaskState.TASK_STATE_RUNNING,
            csimdag.TaskState.TASK_STATE_SCHEDULED,
            csimdag.TaskState.TASK_STATE_RUNNABLE
        ]:
            host = task.hosts[0]
            if host in self._exec_hosts:
                available_cores[host] -= 1
            if task.start_time > 0 and task not in self._started_tasks:
                self._started_tasks.add(task)
                host.data['est'][task] = task.start_time + task.get_eet(host)

        host_ests = {}
        for h in self._exec_hosts:
            host_ests[h] = sorted(h.data['est'].values())[-h.cores:]  # Finding est for (at least) all unfinished tasks

        tasks = simulation.tasks[csimdag.TaskState.TASK_STATE_SCHEDULABLE]
        num_tasks = len(tasks)

        # build ECT matrix
        ECT = np.zeros((num_tasks, len(self._exec_hosts)))
        cached_tasks = {}
        for t, task in enumerate(tasks):
            for h, host in enumerate(self._exec_hosts):
                best_est = host_ests[host][0] if host_ests[host] else 0
                ECT[t][h], cached_tasks[(task, host)] = self.get_ect(best_est, clock, task, host)

        # build schedule
        task_idx = np.arange(num_tasks)
        for _ in range(0, num_tasks):
            min_hosts = np.argmin(ECT, axis=1)
            min_times = ECT[np.arange(ECT.shape[0]), min_hosts]

            if ECT.shape[1] > 1:
                min2_times = np.partition(ECT, 1)[:, 1]
                sufferages = min2_times - min_times
            else:
                sufferages = -min_times

            possible_schedules = []
            for i in range(0, len(task_idx)):
                best_host_idx = int(min_hosts[i])
                best_ect = min_times[i]
                sufferage = sufferages[i]
                possible_schedules.append((i, best_host_idx, best_ect, sufferage))

            t, h, ect = self.batch_heuristic(possible_schedules)
            task = tasks[int(task_idx[t])]
            host = self._exec_hosts[h]

            if host_ests[host]:
                old_est = host_ests[host][0]
                host_ests[host][0] = ect
            else:
                old_est = 0
                host_ests[host].append(ect)
            host_ests[host].sort()

            if available_cores[host]:
                for transfer in cached_tasks[(task, host)]:
                    simulation.remove_dependency(transfer.parents[0], transfer)
                    simulation.remove_dependency(transfer, task)
                    tasks_to_remove.append(transfer)
                task.schedule(host)
                host.data['est'][task] = ect
                available_cores[host] -= 1
                if not any(available_cores.values()):
                    break

            task_idx = np.delete(task_idx, t)
            ECT = np.delete(ECT, t, 0)
            ECT[:,h] += host_ests[host][0] - old_est

        for task in tasks_to_remove:
            simulation.remove_task(task)

    def get_ect(self, est, clock, task, host):
        # In reference implementation here was caching, but it's no longer possible
        parent_connections = [p for p in task.parents if p.kind == csimdag.TaskKind.TASK_KIND_COMM_E2E]
        communications = [self.get_ecomt(conn, conn.parents[0].hosts[0], host) for conn in parent_connections]
        comm_times = [x[0] for x in communications]
        cached_flags = [x[1] for x in communications]
        cached_tasks = [task for (task, is_cached) in zip(parent_connections, cached_flags) if is_cached]
        task_time = (max(comm_times) if comm_times else 0.) + task.get_eet(host)
        return max(est, clock) + task_time, cached_tasks

    def get_ecomt(self, task, host1, host2):
        if host1 == host2:
            return 0, False
        elif (self._data_transfer_mode == scheduler.DataTransferMode.LAZY_CACHING and
              (task.parents[0], host2) in self._cached_tasks):
            return 0, True
        else:
            min_bandwidth = None
            for link in cplatform.route(host1, host2):
                bandwidth = cplatform.link_bandwidth(link) / (self._link_transmissions[link] + 1)
                min_bandwidth = min(bandwidth, min_bandwidth) if min_bandwidth is not None else bandwidth
            return cplatform.route_latency(host1, host2) + task.amount / min_bandwidth, False



class EnhancedBatchMin(EnhancedBatchScheduler):
    """
    Batch-mode MinMin scheduler.

    Schedules all currently schedulable tasks to a best host by ECT in a single batch.

    The order in a batch is determined by a heuristic:
    MinMin prioritizes the tasks with minimum ECT on a best host.
    """

    def batch_heuristic(self, possible_schedules):
        return min(possible_schedules, key=lambda s: (s[2], s[0]))[:-1]


class EnhancedBatchMax(EnhancedBatchScheduler):
    """
    Batch-mode MaxMin scheduler.

    Schedules all currently schedulable tasks to a best host by ECT in a single batch.

    The order in a batch is determined by a heuristic:
    MaxMin prioritizes the tasks with maximum ECT on a best host
    """

    def batch_heuristic(self, possible_schedules):
        return max(possible_schedules, key=lambda s: (s[2], s[0]))[:-1]


class EnhancedBatchSufferage(EnhancedBatchScheduler):
    """
    Batch-mode Sufferage scheduler.

    Schedules all currently schedulable tasks to a best host by ECT in a single batch.

    The order in a batch is determined by a heuristic:
    Sufferage prioritizes tasks with maximum difference between ECT on 2 best hosts
    """

    def batch_heuristic(self, possible_schedules):
        return max(possible_schedules, key=lambda s: (s[3], s[0]))[:-1]
