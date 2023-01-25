#!/bin/bash

TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"

for i1 in 1 2 3 4 5 6 7 8 9
do
  timeout --foreground --signal=SIGINT --kill-after=5m 3.1h \
          python acasxu_repair_1.py --timestamp "$TIMESTAMP" \
          --timeout 3 --specification 2 --network "2,$i1" "$@"
done

# net 3,3 satisfies property 2
for i1 in 1 2 4 5 6 7 8 9
do
  timeout --foreground --signal=SIGINT --kill-after=5m 3.1h \
          python acasxu_repair_1.py --timestamp "$TIMESTAMP" \
          --timeout 3 --specification 2 --network "3,$i1" "$@"
done

# net 4,2 satisfies property 2
for i1 in 1 3 4 5 6 7 8 9
do
  timeout --foreground --signal=SIGINT --kill-after=5m 3.1h \
          python acasxu_repair_1.py --timestamp "$TIMESTAMP" \
          --timeout 3 --specification 2 --network "4,$i1" "$@"
done

for i1 in 1 2 3 4 5 6 7 8 9
do
  timeout --foreground --signal=SIGINT --kill-after=5m 3.1h \
          python acasxu_repair_1.py --timestamp "$TIMESTAMP" \
          --timeout 3 --specification 2 --network "5,$i1" "$@"
done

# timeout --foreground --signal=SIGINT --kill-after=5m 3.1h \
#         python acasxu_repair_1.py --timestamp "$TIMESTAMP" \
#         --timeout 3 --specification 7 --network "1,9" "$@"

# timeout --foreground --signal=SIGINT --kill-after=5m 3.1h \
#         python acasxu_repair_1.py --timestamp "$TIMESTAMP" \
#         --timeout 3 --specification 8 --network "2,9" "$@"

# timeout --foreground --signal=SIGINT --kill-after=5m 3.1h \
#         python acasxu_repair_1.py --timestamp "$TIMESTAMP" \
#         --timeout 3 --specification 1,2,3,4,8 --network "2,9" "$@"
