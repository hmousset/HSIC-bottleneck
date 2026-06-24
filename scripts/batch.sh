#!/bin/bash

# # # # # # # # # # # # # # #
# Batch-running every paper experiment.
#   please see config/README.org for more information
# # # # # # # # # # # # # # #

HERE="$(cd "$(dirname "$0")" && pwd)"

bash "${HERE}/task_general.sh"        # general
bash "${HERE}/task_varied-act.sh"     # fig2a-c
bash "${HERE}/task_varied-depth.sh"   # fig2d-f
bash "${HERE}/task_hsicsolve-beta.sh" # fig3-4
bash "${HERE}/task_varied-ep.sh"      # fig5-ab
bash "${HERE}/task_varied-dim.sh"     # fig6a
bash "${HERE}/task_combsig.sh"        # fig6b
bash "${HERE}/task_resconv.sh"        # fig7
bash "${HERE}/task_needle.sh"         # fig8
