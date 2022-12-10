def get_actionable():
    actionable = {
        'breakable': ('isBroken', [None, step_break]),
        'cookable': ('isCooked', [None, step_cook]),
        'dirtyable': ('isDirty', [step_clean, step_dirty]),
        'toggleable': ('isToggled', [step_turnoff, step_turnon]),
        'openable': ('isOpen', [step_close, step_open]),
        'canBeUsedUp': ('isUsedUp', [None, step_use]),
        'sliceable': ('isSliced', [None, step_slice]),
    }
    return actionable


def action_symmetry():
    paired_set = {('turnon', 'turnoff'), ('clean', 'dirty'), ('close', 'open')}
    paired_dict_1 = {v1: v2 for (v1, v2) in paired_set}
    paired_dict_2 = {v2: v1 for (v1, v2) in paired_set}
    paired_dict = {**paired_dict_1, **paired_dict_2}
    return paired_dict


def step_open(controller, objid):
    return controller.step(
        action="OpenObject",
        objectId=objid,
        openness=0.8,
        forceAction=False
    )


def step_close(controller, objid):
    return controller.step(
        action="CloseObject",
        objectId=objid,
        forceAction=False
    )


def step_dirty(controller, objid):
    return controller.step(
        action="DirtyObject",
        objectId=objid,
        forceAction=False
    )


def step_clean(controller, objid):
    return controller.step(
        action="CleanObject",
        objectId=objid,
        forceAction=False
    )


def step_break(controller, objid):
    return controller.step(
        action="BreakObject",
        objectId=objid,
        forceAction=False
    )


def step_cook(controller, objid):
    return controller.step(
        action="CookObject",
        objectId=objid,
        forceAction=False
    )


def step_slice(controller, objid):
    return controller.step(
        action="SliceObject",
        objectId=objid,
        forceAction=False
    )


def step_turnon(controller, objid):
    return controller.step(
        action="ToggleObjectOn",
        objectId=objid,
        forceAction=False
    )


def step_turnoff(controller, objid):
    return controller.step(
        action="ToggleObjectOff",
        objectId=objid,
        forceAction=False
    )


def step_fill(controller, objid):
    return controller.step(
        action="FillObjectWithLiquid",
        objectId=objid,
        fillLiquid="water",
        forceAction=False
    )


def step_empty(controller, objid):
    return controller.step(
        action="EmptyLiquidFromObject",
        objectId=objid,
        forceAction=False
    )


def step_use(controller, objid):
    return controller.step(
        action="UseUpObject",
        objectId=objid,
        forceAction=False
    )
