package network;

public class Connection {
    private double weight;
    private Neuron targetNeuron;

    public Connection(Neuron targetNeuron) {
        this.weight = 1; // initial weight for network
        this.targetNeuron = targetNeuron;
    }

    public double getWeight() {
        return weight;
    }

    public Neuron getTargetNeuron() {
        return targetNeuron;
    }
}
