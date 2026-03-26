package me.damoebe.architectures;

/**
 * An interface for backpropagation implemented by each module that supports delta-gradient based weight updates.
 */
public interface Backpropagation {
    /**
     * The method in which the backpropagation process should be managed.
     * @param deltas The deltas from the previous backpropagation step, which are being passed into this module
     * @return The new deltas used for the next backpropagation step.
     */
    double[][] backPropagate(double[][] deltas);
}
