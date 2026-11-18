#!/bin/bash
# Start ACE as the kernel's first choice for the OOM killer, so that if several
# parallel ACE jobs peak at once, one ACE process is killed (and its item is
# recorded as a crash and retried) instead of anything else on the machine.
echo 1000 > /proc/self/oom_score_adj
exec "${ERG_ACE:-$HOME/delphin/tools/ACE/ace-0.9.34/ace}" "$@"
