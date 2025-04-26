import numpy as np

def step_ECA(state, rule):
    '''
    Compute the next state of the ECA with periodic boundary conditions

    Param: state (1D array): the state of the ECA
    Param: rule (1D array of 8 elements): the rule of the ECA in binary form (inverse of binary used by Wolfram, first 000, then 001, ...)

    Return: 1D array: the next state of the ECA
    '''
    state_size=state.size
    new_state=np.zeros(state_size, dtype=np.int8)
    for i in range(state_size):
        left=state[(i-1)%state_size]
        middle=state[i]
        right=state[(i+1)%state_size]
        if left==0 and middle==0 and right==0:
            new_state[i]=rule[0]
        elif left==0 and middle==0 and right==1:
            new_state[i]=rule[1]
        elif left==0 and middle==1 and right==0:
            new_state[i]=rule[2]
        elif left==0 and middle==1 and right==1:
            new_state[i]=rule[3]
        elif left==1 and middle==0 and right==0:
            new_state[i]=rule[4]
        elif left==1 and middle==0 and right==1:
            new_state[i]=rule[5]   
        elif left==1 and middle==1 and right==0:
            new_state[i]=rule[6]
        elif left==1 and middle==1 and right==1:
            new_state[i]=rule[7]
    return new_state

def dynamics_ECA(initial_state, rule, n_steps=20):
    '''
    Compute the dynamics of the ECA.

    Param: initial_state (1D array): the initial state of the ECA
    Param: rule (1D array of 8 elements): the rule of the ECA in binary form (inverse of binary used by Wolfram, first 000, then 001, ...)
    Param: n_steps (int): the number of time steps

    Return: 2D array: the states of the ECA at each time step
    '''
    states=np.zeros((n_steps+1, initial_state.size))
    states[0,:]=initial_state
    for i in range(n_steps):
        states[i+1,:]=step_ECA(states[i,:], rule)
    return states