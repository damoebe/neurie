package me.damoebe.architectures.transformer;

import me.damoebe.architectures.Network;
import me.damoebe.architectures.mlp.FFNetwork;
import me.damoebe.architectures.transformer.mha.Head;

import java.util.ArrayList;
import java.util.List;

/**
 * The class for a Decoder-Only Transformer, used for generative tasks
 * @param <N> The feed-forward neural network type used for this transformer
 */
public class DecoderTransformer<N extends FFNetwork> extends Network {
    /**
     * A list of all used decoder blocks for this Transformer
     */
    private final List<Block<Head, N>> decoderBlocks = new ArrayList<>();

    private List<Double> currentInput = new ArrayList<>();
    private List<Double> currentOutput = new ArrayList<>();

    /**
     * Main constructor for a Decoder-Only Transformer with a reference Decoder-Block
     * @param decoderBlockAmount The amount of Decoder Blocks that should be used.
     * @param referenceBlock The reference Decoder-Block that will be used to construct the blocks
     */
    public DecoderTransformer(int decoderBlockAmount, Block<Head, N> referenceBlock){
        for (int decoderBlockIndex = 0; decoderBlockIndex != decoderBlockAmount; decoderBlockIndex++){
            decoderBlocks.add(referenceBlock.clone());
        }
    }

    @Override
    public void train(List<Double> input, List<Double> optimalOutput) {

    }

    @Override
    public List<Double> getOutput() {
        return currentOutput;
    }

    @Override
    public void insertInput(List<Double> input) {
        this.currentInput = input;
    }

    /**
     * Updates All activations in the transformer as well as the currentOutput
     */
    @Override
    public void updateAllActivations() {

    }
}
