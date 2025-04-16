package network;

import java.util.ArrayList;
import java.util.List;

public class Network {

    private final List<Layer> layers = new ArrayList<>(); // the whole network in a list

    public Network(int inputSize, int outputSize, int hiddenLayerSize, int hiddenLayerAmount){ // adjustable values
        // generate network
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
            neurons.add(new Neuron(connections));
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

    public double train(List<Double> input, List<Double> optimalOutput){
        return 0; // train network by updating optimalActivations and weights
    }

    private void updateOptimalActivations(){
        return;
    }

    private void updateAllWeights(){
        return;
    }
}
