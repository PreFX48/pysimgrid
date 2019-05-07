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

from ... import cscheduling
from .. import scheduler


class EnhancedLookahead(scheduler.StaticScheduler):
  def get_schedule(self, simulation):
    """
    Overriden.
    """
    nxgraph = simulation.get_task_graph()
    platform_model = cscheduling.PlatformModel(simulation)
    state = cscheduling.SchedulerState(simulation)
    data_transfer_mode = self._data_transfer_mode.name

    ordered_tasks = cscheduling.heft_order(nxgraph, platform_model)
    for idx, task in enumerate(ordered_tasks):
      if cscheduling.try_schedule_boundary_task(task, nxgraph, platform_model, state):
        continue
      current_min = cscheduling.MinSelector()
      for host, timesheet in state.timetable.items():
        if cscheduling.is_master_host(host):
          continue
        temp_state = state.copy()
        if data_transfer_mode == 'EAGER_CACHING':
          est, cached_tasks, transfer_finishes = platform_model.enhanced_est(host, dict(nxgraph.pred[task]), state, True)
        elif data_transfer_mode == 'EAGER':
          est, cached_tasks, transfer_finishes = platform_model.enhanced_est(host, dict(nxgraph.pred[task]), state, False)
        else:
          raise NotImplementedError('Enhanced Lookahead cannot yet work in {} mode'.format(data_transfer_mode))
        eet = platform_model.eet(task, host)
        pos, start, finish = cscheduling.timesheet_insertion(timesheet, host.cores, est, eet)
        if data_transfer_mode == 'EAGER_CACHING':
          for parent, edge in nxgraph.pred[task].items():
            transfer_start_time = temp_state.task_states[parent]['ect']
            transfer_src_host = temp_state.task_states[parent]['host']
            transfer_task = temp_state.transfer_task_by_name[edge['name']]
            temp_state.cached_transfers.setdefault(transfer_task.parents[0].name, {})
            cached_hosts = temp_state.cached_transfers[transfer_task.parents[0].name]
            if edge['name'] not in cached_tasks:
              if not cached_hosts.get(host.name) or cached_hosts[host.name][0] > transfer_finishes[transfer_task.name]:
                cached_hosts[host.name] = (transfer_finishes[transfer_task.name], transfer_task)
              temp_state.update_schedule_for_transfer(transfer_task, transfer_start_time, transfer_src_host, host)
        else:
          for parent, edge in nxgraph.pred[task].items():
            transfer_start_time = temp_state.task_states[parent]['ect']
            transfer_src_host = temp_state.task_states[parent]['host']
            transfer_task = temp_state.transfer_task_by_name[edge['name']]
            temp_state.update_schedule_for_transfer(transfer_task, transfer_start_time, transfer_src_host, host)
        temp_state.update(task, host, pos, start, finish)
        cscheduling.enhanced_heft_schedule(simulation, nxgraph, platform_model, temp_state, ordered_tasks[(idx + 1):],
                                  self._data_transfer_mode.name, 1)
        total_time = max([state["ect"] for state in temp_state.task_states.values()])
        # key order to ensure stable sorting:
        #  first sort by HEFT makespan (as Lookahead requires)
        #  if equal - sort by host speed
        #  if equal - sort by host name (guaranteed to be unique)
        current_min.update((total_time, host.speed, host.name), (host, pos, start, finish, cached_tasks, transfer_finishes))
      host, pos, start, finish, cached_tasks, transfer_finishes = current_min.value
      if data_transfer_mode == 'EAGER_CACHING':
        for parent, edge in nxgraph.pred[task].items():
          transfer_start_time = state.task_states[parent]['ect']
          transfer_src_host = state.task_states[parent]['host']
          transfer_task = state.transfer_task_by_name[edge['name']]
          state.cached_transfers.setdefault(transfer_task.parents[0].name, {})
          cached_hosts = state.cached_transfers[transfer_task.parents[0].name]
          if edge['name'] in cached_tasks:
            _, old_transfer = state.cached_transfers[parent.name][host.name]
            assert len(transfer_task.parents) == 1
            del state.transfer_tasks[transfer_task]
            del state.transfer_task_by_name[transfer_task.name]
            assert old_transfer != transfer_task, 'Task {} tries to replace {} with itself on host {}'.format(task.name, transfer_task.name, host.name)
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
            state.update_schedule_for_transfer(transfer_task, transfer_start_time, transfer_src_host, host)
      else:
        for parent, edge in nxgraph.pred[task].items():
          transfer_start_time = state.task_states[parent]['ect']
          transfer_src_host = state.task_states[parent]['host']
          transfer_task = state.transfer_task_by_name[edge['name']]
          state.update_schedule_for_transfer(transfer_task, transfer_start_time, transfer_src_host, host)
      state.update(task, host, pos, start, finish)

    # store ECT in tasks for QUEUE_ECT data transfer mode
    for task, task_state in state.task_states.items():
      if task.data is None:
        task.data = {}
      task.data['ect'] = task_state["ect"]
    expected_makespan = max([state["ect"] for state in state.task_states.values()])
    return state.schedule, expected_makespan
