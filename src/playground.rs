use ndarray::{Array2, s};

struct LayerDense {
    weights: Array2<f32>,
    biases: Array2<f32>,
}

impl LayerDense {
    fn new(n_inputs: usize, n_neurons: usize) -> Self {
        let weights = Array2::zeros((n_inputs, n_neurons));
        let biases = Array2::zeros((1, n_neurons));
        Self { weights, biases }
    }

    fn forward(&mut self, inputs: &Array2<f32>) -> Array2<f32> {
        self.weights.dot(inputs) + &self.biases
    }
}

struct ActivationRelu;

impl ActivationRelu {
    fn forward(inputs: &Array2<f32>) -> Array2<f32> {
        inputs.mapv(|x| x.max(0.0))
    }
}

struct ActivationSoftmax;

impl ActivationSoftmax {
    fn forward(inputs: &Array2<f32>) -> Array2<f32> {
        let exp_values = inputs.mapv(|x| x.exp() - inputs.max(ndarray::Axis(1)));
        let sum = exp_values.sum();
        exp_values.mapv(|x| x / sum)
    }
}
struct Loss;

impl Loss {
    fn calculate(&self, output: &Array2<f32>, y: &Array2<f32>) -> f32 {
        let sample_losses = self.forward(output, y);
        sample_losses.mean()
    }
}
struct LossCategoricalCrossEntropy<Loss>;

impl LossCategoricalCrossEntropy<Loss> {
    fn forward(&self, y_pred: &Array2<f32>, y_true: &Array2<f32>) -> Array2<f32> {
        let samples = y_pred.shape()[0];
        let y_pred_clipped = y_pred.mapv(|x| x.max(1e-7).min(1.0 - 1e-7));

        let correct_confidences: Array2<f32>;
        if y_true.shape()[1] == 1 {
            correct_confidences = y_pred_clipped.slice(s![.., y_true.slice(s![.., 0])]);
        } else {
            correct_confidences = y_pred_clipped * y_true;
            correct_confidences = correct_confidences.sum_axis(ndarray::Axis(1));
        }

        let negative_log_likelihoods = correct_confidences.mapv(|x| -x.ln());
        negative_log_likelihoods
    }
}





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

