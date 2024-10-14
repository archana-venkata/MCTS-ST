rollouts = [[(0, 'event1'), (0, 'event3')],
            [(2, 'event3'), (0, 'event3'), (0, None), (3, 'event3')],
            [(3, 'event4')],
            [(0, 'event1'), (2, 'event3'), (2, None), (2, 'event4')],
            ]  # rollout trajectory of action and events pairs
strategies = [['event1', 'event2'], ['event3', 'event4']]


def hasEvent(element, event):
    indices = [i for i, v in enumerate(element) if v[1] == event]
    if len(indices) > 0:
        return True
    return False


def calc_action_uncertainties(action, event, possible_next_actions):
    # get all strategies containing 'event'
    for strategy in [x for x in strategies if event in x]:
        index = strategy.index(event)
        next_event = strategy[index+1]
        print(next_event)
        # get all rollouts containing the next event in the strategy 'next_event'
        subset_rollouts = [rollout for rollout in rollouts if hasEvent(rollout, next_event)]
        if len(subset_rollouts) == 0:
            alt_subset_rollouts = [rollout for rollout in rollouts if hasEvent(rollout, event)]
            for rollout in alt_subset_rollouts:
                event_indices = [i for i, v in enumerate(rollout) if v[1] == event]
                if len(event_indices) > 0:
                    next_action = rollout[event_indices[0]+1][0]
                    possible_next_actions[next_action] -= 0.1
            return possible_next_actions

        for rollout in subset_rollouts:
            event_indices = [i for i, v in enumerate(rollout) if v[1] == event]
            if len(event_indices) > 0:
                print(f"next action: {rollout[event_indices[0]+1][0]}")
                next_action = rollout[event_indices[0]+1][0]
                possible_next_actions[next_action] += 0.1
    return possible_next_actions


print(calc_action_uncertainties(0, 'event1', [0.25, 0.25, 0.25, 0.25]))
