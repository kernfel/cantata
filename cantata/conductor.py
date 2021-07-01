import torch
from box import Box

import cantata.elements as ce
from cantata import init

def assemble_synapse(conf_pre, batch_size, dt, conf_post = None, name_post = None,
        Synapse = ce.Synapse, Current = ce.SynCurrent, STP = ce.STP, LTP = ce.Abbott,
        **kwargs
    ):
    if conf_post is None:
        conf_post = conf_pre
    sub = dict(stp=None, ltp=None, current=None)
    projections = init.build_projections(conf_pre, conf_post, name_post)

    if LTP is not None:
        nPre = init.get_N(conf_pre)
        nPost = init.get_N(conf_post)
        sub['ltp'] = LTP(projections, conf_post, batch_size, nPre, nPost, dt)

    if STP is not None:
        n_i_delays = len(init.get_delays(conf_pre, dt, conf_post is not conf_pre))
        sub['stp'] = STP(conf_pre, n_i_delays, batch_size, dt)

    if Current is not None:
        sub['current'] = Current(conf_post, batch_size, dt)

    return Synapse(projections, conf_pre, conf_post, batch_size, dt, **sub, **kwargs)

def assemble(conf, batch_size, dt, out_dtype = torch.float,
        Input = ce.PoissonInput,
        Circuit = ce.SNN, Membrane = ce.Membrane, Spikes = ce.ALIFSpikes,
        InputSynapse = assemble_synapse, CircuitSynapse = assemble_synapse,
        **kwargs
    ):
    input = Input(conf.input, batch_size, dt)

    assert len(conf.areas) == 1, 'assemble() does not support distinct areas.'
    name = list(conf.areas.keys())[0]
    cconf = conf.areas[name]

    membrane = Membrane(cconf, batch_size, dt)
    spikes = Spikes(cconf, batch_size, dt)
    csyn = CircuitSynapse(cconf, batch_size, dt, **kwargs)
    isyn = InputSynapse(conf.input, batch_size, dt, cconf, name, **kwargs)

    circuit = Circuit(cconf, batch_size, dt, membrane, spikes, isyn, csyn)

    return ConcertMaster(input, circuit, **kwargs)

Conductor = assemble # For backward compatibility

class ConcertMaster(torch.nn.Module):
    '''
    Putting it all together.
    Input: Rates
    Output: Output spikes
    '''
    def __init__(self, input, circuit, out_dtype = torch.float, **kwargs):
        super(ConcertMaster, self).__init__()

        self.input = input
        self.circuit = circuit
        self.out_dtype = out_dtype

        self.reset()

    def reset(self):
        self.circuit.reset()

    def forward(self, rates):
        '''
        rates: (t, batch, channels) in Hz
        Output: Tuple of (input, output) spikes as (t, batch, N) tensors
        '''
        batch_size = rates.shape[1]
        T = len(rates)
        inputs = torch.empty(T, batch_size, self.input.N, dtype=self.out_dtype)
        outputs = torch.empty(T, batch_size, self.circuit.N, dtype=self.out_dtype)

        # Main loop
        for t, rate_t in enumerate(rates):
            Xi = self.input(rate_t)
            inputs[t] = Xi[0]
            outputs[t] = self.circuit(Xi)

        return inputs, outputs
