package me.damoebe.architectures.transformer;

import me.damoebe.architectures.Network;

import java.util.List;

public class S2STransformer extends Network {
    @Override
    public void train(List<Double> input, List<Double> optimalOutput) {

    }

    @Override
    public List<Double> getOutput() {
        return List.of();
    }

    @Override
    public void insertInput(List<Double> input) {

    }

    @Override
    public void updateAllActivations() {

    }
}
