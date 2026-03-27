from vmas.simulator.core import World
from vmas.simulator.joints import Joint


class SalpNavigateWorld(World):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def attach_joint(self, joint: Joint):
        for constraint in joint.joint_constraints:
            self._joints.update(
                {
                    frozenset(
                        {constraint.entity_a.name, constraint.entity_b.name}
                    ): constraint
                }
            )

    def detach_joint(self, joint: Joint):
        for constraint in joint.joint_constraints:
            self._joints.update(
                {frozenset({constraint.entity_a.name, constraint.entity_b.name}): None}
            )
