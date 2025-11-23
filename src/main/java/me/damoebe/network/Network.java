package me.damoebe.network;

import java.util.ArrayList;
import java.util.List;

public abstract class Network {

    // the whole network in a list
    final List<Layer> layers = new ArrayList<>();
    double loss = 0;
    double noise = 0.1;

    public Network(int inputSize, int outputSize, int hiddenLayerSize, int hiddenLayerAmount){ // adjustable values
        // generate network
        Layer inputLayer = generateLayer(inputSize, null, 0);
        layers.add(inputLayer);
        Layer prevLayer = inputLayer;
        for (int i = 0; i != hiddenLayerAmount; i++){
            Layer hiddenLayer = generateLayer(hiddenLayerSize, prevLayer, 0);
            layers.add(hiddenLayer);
            prevLayer = hiddenLayer;
        }
        Layer outputLayer = generateLayer(outputSize, prevLayer, 0);
        layers.add(outputLayer);
    }

    public Network(int inputSize, int outputSize, int hiddenLayerSize, int hiddenLayerAmount, double learningRate){ // adjustable values
        // generate network
        Layer inputLayer = generateLayer(inputSize, null, learningRate);
        layers.add(inputLayer);
        Layer prevLayer = inputLayer;
        for (int i = 0; i != hiddenLayerAmount; i++){
            Layer hiddenLayer = generateLayer(hiddenLayerSize, prevLayer, learningRate);
            layers.add(hiddenLayer);
            prevLayer = hiddenLayer;
        }
        Layer outputLayer = generateLayer(outputSize, prevLayer, learningRate);
        layers.add(outputLayer);
    }

    // generates a layer for the network
    private Layer generateLayer(int size, Layer prevLayer, double learningRate){
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
            layers.get(0).neurons().get(i).setActivation(input.get(i));
        }
    }

    // update network loss
    void updateLoss(List<Double> optimalOutput){
        int i = 0;
        double totalLoss = 0;
        for (double output : getOutput()){
            double error = output - optimalOutput.get(i);
            totalLoss += error * error;
            i++;
        }
        loss = totalLoss / optimalOutput.size();
    }

    // returns the activations of the last layer
    public List<Double> getOutput(){
        List<Double> output = new ArrayList<>();
        for (Neuron neuron : layers.get(layers.size()-1).neurons()){
            output.add(neuron.getActivation());
        }
        return output;
    }

    public void setNoise(double noise){
        this.noise = noise;
    }

    public double getLoss(){
        return loss;
    }

    public abstract void train(List<Double> input, List<Double> optimalOutput);
    public abstract void finishEpoch();
    public abstract double getNetworkLoss();
}
