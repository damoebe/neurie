package network;

import java.util.ArrayList;
import java.util.List;

public class Network {

    private final List<Layer> layers = new ArrayList<>(); // the whole network in a list

    private final double learningRate;

    public Network(int inputSize, int outputSize, int hiddenLayerSize, int hiddenLayerAmount, double learningRate){ // adjustable values
        // generate network
        this.learningRate = learningRate;
        Layer inputLayer = generateLayer(inputSize, null);
        layers.add(inputLayer);
        Layer prevLayer = inputLayer;
        for (int i = 0; i != hiddenLayerAmount; i++){
            Layer hiddenLayer = generateLayer(hiddenLayerSize, prevLayer);
            layers.add(hiddenLayer);
            prevLayer = hiddenLayer;
        }
        Layer outputLayer = generateLayer(outputSize, prevLayer);
        layers.add(outputLayer);
    }

    // generates a layer for the network
    private Layer generateLayer(int size, Layer prevLayer){
        List<Connection> connections = new ArrayList<>(); // neurons from one layer have same dependencies
        if (prevLayer != null){ // if not input layer
            for (Neuron neuron : prevLayer.neurons()){
                connections.add(new Connection(neuron));
            }
        }
        List<Neuron> neurons = new ArrayList<>();
        for (int i = 0; i != size; i++){
            neurons.add(new Neuron(new ArrayList<>(connections), learningRate));
        }
        return new Layer(neurons); // returns record
    }

    // updates all activations
    public void updateAllActivations(){
        for (Layer layer : layers){
            for (Neuron neuron : layer.neurons()){
                neuron.updateActivation();
            }
        }
    }

    // sets the input neurons activations
    public void insertInput(List<Double> input){
        for (int i = 0; i != input.size(); i++){
            layers.getFirst().neurons().get(i).setActivation(input.get(i));
        }
    }

    // returns the activations of the last layer
    public List<Double> getOutput(){
        List<Double> output = new ArrayList<>();
        for (Neuron neuron : layers.getLast().neurons()){
            output.add(neuron.getActivation());
        }
        return output;
    }

    // uses backpropagation to update the weights with ONE set of date
    public void train(List<Double> input, List<Double> optimalOutput){
         insertInput(input);
         updateAllActivations();
         updateOutputDeltas(optimalOutput);
         updateHiddenDeltas();
         updateAllWeightsAndBiases();
    }

    // update all deltas in the network
    private void updateHiddenDeltas() {
        for (int l = layers.size() - 2; l > 0; l--) { // skip output and input layer
            for (Neuron neuron : layers.get(l).neurons()) {
                double output = neuron.getActivation();
                double sum = 0.0;
                for (Neuron next : layers.get(l + 1).neurons()) {
                    for (Connection conn : next.getConnections()) {
                        if (conn.getSourceNeuron() == neuron) {
                            sum += conn.getWeight() * next.getDelta();
                        }
                    }
                }
                double delta = output * (1 - output) * sum;
                neuron.setDelta(delta);
            }
        }
    }

    // update all output deltas
    private void updateOutputDeltas(List<Double> targetOutputs) {
        List<Neuron> outputNeurons = layers.getLast().neurons();
        for (int i = 0; i < outputNeurons.size(); i++) {
            Neuron neuron = outputNeurons.get(i);
            double output = neuron.getActivation();
            double error = output - targetOutputs.get(i); // calc error
            double delta = error * output * (1 - output); // delta learning rule
            neuron.setDelta(delta);
        }
    }

    // updates all connection-weights and biases within the network
    private void updateAllWeightsAndBiases() {
        for (Layer layer : layers) {
            for (Neuron neuron : layer.neurons()) {
                neuron.updateWeights();
                neuron.updateBias();
            }
        }
    }
}