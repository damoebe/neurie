package me.damoebe.architectures.mlp;

import com.google.gson.Gson;
import com.google.gson.stream.JsonReader;
import me.damoebe.architectures.Network;
import me.damoebe.architectures.mlp.structure.Connection;
import me.damoebe.architectures.mlp.structure.Layer;
import me.damoebe.architectures.mlp.structure.Neuron;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;

/**
 * The abstract class for FFNeuralNetworks
 */
public abstract class FFNetwork extends Network {

    /**
     * The whole network in a list
     */
    protected final List<Layer> layers = new ArrayList<>();
    /**
     * The current loss of the network
     */
    protected double currentLoss = 0;
    /**
     * The configured noise of the network
     */
    protected double noise = 0.1;
    protected final double learningRate ;


    /**
     * The main constructor of a FFNetwork
     * @param inputSize The size of the first layer
     * @param outputSize The size of the last layer
     * @param hiddenLayerSize The size of the hiddenlayers
     * @param hiddenLayerAmount The amount of hiddenlayers that should be generated
     */
    protected FFNetwork(int inputSize, int outputSize, int hiddenLayerSize, int hiddenLayerAmount, double learningRate){ // adjustable values
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

    /**
     * FFNetwork constructor for networks without a learning rate - 0.0 (for example evolution learning)
     * @param inputSize The size of the first layer
     * @param outputSize The size of the last layer
     * @param hiddenLayerSize The size of the hiddenlayers
     * @param hiddenLayerAmount The amount of hiddenlayers that should be generated
     */
    protected FFNetwork(int inputSize, int outputSize, int hiddenLayerSize, int hiddenLayerAmount){ // adjustable values
        this(inputSize, outputSize, hiddenLayerSize, hiddenLayerAmount, 0);
    }

    /**
     * A method for cloning a FFNetwork object
     * @return the cloned network object
     */
    public abstract FFNetwork clone();

    /**
     * Generates a new layer for the FFNetwork constructor
     * @param size The amount of neurons that layer should contain
     * @param prevLayer The previous Layer in the network (null if there is no previous layer)
     * @return A new Layer that points to the previous layer
     */
    public static Layer generateLayer(int size, Layer prevLayer){
        List<Connection> connections = new ArrayList<>(); // neurons from one layer have same dependencies
        if (prevLayer != null){ // if not input layer
            for (Neuron neuron : prevLayer.neurons()){
                connections.add(new Connection(neuron));
            }
        }
        List<Neuron> neurons = new ArrayList<>();
        for (int i = 0; i != size; i++){
            // clone connections list to avoid object pointing
            neurons.add(new Neuron(new ArrayList<>(connections)));
        }
        return new Layer(neurons); // returns record
    }

    /**
     * Calls the updatedActivation() function for each neuron in the network
     */
    public void updateAllActivations(){
        for (Layer layer : layers){
            for (Neuron neuron : layer.neurons()){
                neuron.updateActivation();
            }
        }
    }

    /**
     * Inserts input activations into the network
     * @param input A Double ArrayList containing the neuron activations.
     */
    public void insertInput(List<Double> input){
        for (int i = 0; i != input.size(); i++){
            layers.getFirst().neurons().get(i).setActivation(input.get(i));
        }
    }

    /**
     * Updates the network loss based on a targetActivation list
     * @param optimalOutput The target activations for the current repetition
     */
    public void updateLoss(List<Double> optimalOutput){
        int i = 0;
        double totalLoss = 0;
        for (double output : getOutput()){
            double error = output - optimalOutput.get(i);
            totalLoss += error * error;
            i++;
        }
        currentLoss = totalLoss / optimalOutput.size();
    }

    /**
     * Gets all output activations
     * @return A Double List containing all output layer activations.
     */
    public List<Double> getOutput(){
        List<Double> output = new ArrayList<>();
        for (Neuron neuron : layers.get(layers.size()-1).neurons()){
            output.add(neuron.getActivation());
        }
        return output;
    }

    /**
     * Sets the network noise.
     * @param noise The new noise of the network.
     */
    public void setNoise(double noise){
        this.noise = noise;
    }

    /**
     * A Getter for the size of the first layer's size
     * @return The input-layer size
     */
    public int getInputSize(){
        return this.layers.getFirst().neurons().size();
    }

    /**
     * A Getter for the size of the last layer's size
     * @return The output-layer size
     */
    public int getOutputSize(){
        return this.layers.getLast().neurons().size();
    }

    /**
     * A Getter for the size of the middle layers size
     * @return The size of the hidden layers
     */
    public int getHiddenLayerSize(){
        return this.layers.get(1).neurons().size();
    }

    /**
     * A Getter for the amount of hidden-layers in this network
     * @return The hidden-layer amount
     */
    public int getHiddenLayerAmount(){
        return this.layers.size()-2;
    }

    /**
     * The abstract train function
     * @param input The inputs for this repetition
     * @param optimalOutput The optimalOutputs for this repetition
     */
    public abstract void train(List<Double> input, List<Double> optimalOutput);


    /**
     * Gets the network loss
     * @return The network loss double
     */
    public abstract double getNetworkLoss();

    /**
     * Getter for the layer-list of this network
     * @return The layers in a list
     */
    public List<Layer> getLayers(){
        return this.layers;
    }
}
