use ndarray::{Array2, s, ArrayBase, OwnedRepr, Dim};

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
struct Loss{
    base: LossCategoricalCrossEntropy,
}

impl Loss {
    fn new() -> Self {
        Self { base: LossCategoricalCrossEntropy }
    }
    fn calculate(&self, output: &[f64], y: &[f64]) -> f64 {
        let sample_losses = self.LossCategoricalCrossEntropy.forward(output, y);
        let data_loss = sample_losses.iter().sum::<f64>() / sample_losses.len() as f64;
        data_loss
    }
}

struct LossCategoricalCrossEntropy;

impl LossCategoricalCrossEntropy {
  

    fn forward(&self, y_pred: &[f64], y_true: &[f64]) -> Vec<f64> {
        let samples = y_pred.len();
        let y_pred_clipped = y_pred.iter().map(|y| y.max(1e-7).min(1.0 - 1e-7)).collect::<Vec<f64>>();

        let correct_confidences: Vec<f64> = if y_true.len() == samples {
            y_pred_clipped
                .iter()
                .enumerate()
                .filter_map(|(i, y)| if i as i64 == y_true[i] as i64 { Some(*y) } else { None })
                .collect()
        } else {
            y_pred_clipped
                .iter()
                .zip(y_true.chunks(samples))
                .map(|(y_pred, y_true)| y_pred * y_true.iter().sum::<f64>())
                .collect()
        };

        let negative_log_likelihoods: Vec<f64> = correct_confidences
            .iter()
            .map(|y| (-y.ln()).max(f64::EPSILON))
            .collect();
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

