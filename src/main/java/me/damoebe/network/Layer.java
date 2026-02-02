package me.damoebe.network;

import java.util.List;

/**
 * Record class for Neurons
 * @param neurons The layers neurons
 */
public record Layer(List<Neuron> neurons) {
}
