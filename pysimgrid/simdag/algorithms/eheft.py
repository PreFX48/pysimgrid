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

from .. import scheduler
from ... import scheduling as cscheduling
# from ... import cscheduling


class EHEFT(scheduler.StaticScheduler):
  """
  Enhanced Heterogeneous Earliest Finish Time (EHEFT) scheduler which takes into account network contention.
  Should be much slower than original HEFT, partly because of the lack of Cython optimizations, partly algorithmically
  """

  def get_schedule(self, simulation):
    nxgraph = simulation.get_task_graph()
    platform_model = cscheduling.PlatformModel(simulation)
    state = cscheduling.SchedulerState(simulation)

    ordered_tasks = cscheduling.heft_order(nxgraph, platform_model)

    cscheduling.enhanced_heft_schedule(simulation, nxgraph, platform_model, state, ordered_tasks, self._data_transfer_mode.name, 0)
    # store ECT in tasks for QUEUE_ECT data transfer mode
    for task, task_state in state.task_states.items():
      if task.data is None:
        task.data = {}
      task.data['ect'] = task_state["ect"]
    expected_makespan = max([state["ect"] for state in state.task_states.values()])
    return state.schedule, expected_makespan
