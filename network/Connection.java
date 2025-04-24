package network;

public class Connection {
    private double weight;
    private final Neuron sourceNeuron;

    public Connection(Neuron targetNeuron) {
        this.weight = 1; // initial weight for network
        this.sourceNeuron = targetNeuron;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight){
        this.weight = weight;
    }

    public Neuron getSourceNeuron() {
        return sourceNeuron;
    }
}