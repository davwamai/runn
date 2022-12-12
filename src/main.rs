fn main() {
    let inputs = [1.0, 2.0, 3.0, 2.5];


    let weights = [[0.2, 0.8, -0.5, 1.0],
                [0.5, -0.91, 0.26, -0.5],
                [-0.26, -0.27, 0.17, 0.87]];

   
    let biases = [2.0, 3.0, 0.5];

    let mut layer_outputs = Vec::new();

    let iter = weights.iter().zip(biases.iter());

    for it in iter{
        let (neuron_weight, neuron_bias) = it;
        let mut neuron_output = 0.0;
        let inp_and_weights = inputs.iter().zip(neuron_weight.iter());
        for _inpw_iter in inp_and_weights{
            let (inp, wt) = _inpw_iter;
            neuron_output += inp*wt;    
        }
        neuron_output += neuron_bias;
        layer_outputs.push(neuron_output);
    }

    // for n in layer_outputs{
    //     println!("{}", n);
    // }
        println!("{:#?}",layer_outputs);

    
}