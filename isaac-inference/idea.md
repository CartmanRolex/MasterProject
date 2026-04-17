Two skills
"Grasp orange X" (X = left, middle, right)

Drops whatever is in the gripper if anything
Approaches orange X
Grasps it
Goes idle holding (or idle empty if grasp failed)

"Put into plate"

Transports current orange to plate
Releases
Goes idle empty

Interruption rule
"Grasp orange X" can be issued at any time, including mid-place. When it is, the robot drops whatever it holds and starts the approach. This is the universal recovery primitive.
Training data
Random sequences of these two skills where:

"Grasp orange X" can interrupt a "Put into plate" at any point mid-trajectory
X is chosen randomly, including redundant re-grasps of already-placed oranges or empty positions
Starting positions for "Grasp orange X" are highly varied — neutral, mid-air, above plate, etc.

Orchestrator logic at inference
The orchestrator just watches the scene and decides:

Orange not in plate → issue "Grasp orange X" for the right orange → then "Put into plate"
Orange fell during place → issue "Grasp orange X" again
Orange inaccessible → issue "Grasp orange Y" for a different one