#!/bin/bash

for i in {1..100}
do
python IoRLO_main.py
mv ep_reward.txt $i.txt
done
