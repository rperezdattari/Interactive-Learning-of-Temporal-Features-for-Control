from agents.DCOACH import DCOACH


#  Select agent
def agent_selector(agent_type, config_agent):

    if agent_type == 'DCOACH':
        return DCOACH(dim_a=config_agent.getint('dim_a'),
                      action_upper_limits=config_agent['action_upper_limits'],
                      action_lower_limits=config_agent['action_lower_limits'],
                      e=config_agent['e'],
                      buffer_min_size=config_agent.getint('buffer_min_size'),
                      buffer_max_size=config_agent.getint('buffer_max_size'),
                      buffer_sampling_rate=config_agent.getint('buffer_sampling_rate'),
                      buffer_sampling_size=config_agent.getint('buffer_sampling_size'))

    else:
        raise NameError('Not valid network.')
