package me.damoebe.architectures.transformer.mha;

import me.damoebe.architectures.transformer.embedding.Matrix;
import me.damoebe.architectures.transformer.embedding.Sequence;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The Encoder-Decoder Head class
 */
public class EDHead extends Head{

    /**
     * The main constructor of the decoder Head
     * Identical to the super constructor
     *
     * @param inputEmbeddingAmount The amount of input embeddings
     * @param inputEmbeddingSize   The size of each input embedding
     */
    public EDHead(int inputEmbeddingAmount, int inputEmbeddingSize) {
        super(inputEmbeddingAmount, inputEmbeddingSize, false); // no mask for decoder-encoder heads
    }

    /**
     * Calculates the QKVMatrices based on the decoder and encoder input embeddings.
     * @param inputEmbeddings The input Embedding lists -> here it has to contain 2 embedding lists. [0] is the decoder
     *                        input and [1] is the encoder input
     * @return The query, key and value matrices which will be used to calculate the attention.
     */
    @Override
    protected List<double[][]> generateQKVMatrices(Sequence[] inputEmbeddings) throws Exception{
        if (inputEmbeddings.length != 2) throw new Exception("The EDHead class has to be provided with exactly 2 inputs!");
        List<double[][]> QKV = new ArrayList<>();

        // bad performance fix in future
        // queries come from decoder, keys and values from encoder
        double[][] queries = Matrix.add(
                Objects.requireNonNull(Matrix.multiply(inputEmbeddings[0].getVerticalMatrix(), weights.getFirst())),
                biases.getFirst()
        );

        double[][] keys = Matrix.add(
                Objects.requireNonNull(Matrix.multiply(inputEmbeddings[1].getVerticalMatrix(), weights.get(1))),
                biases.get(1)
        );

        double[][] values = Matrix.add(
                Objects.requireNonNull(Matrix.multiply(inputEmbeddings[1].getVerticalMatrix(), weights.getLast())),
                biases.getLast()
        );

        assert queries != null && keys != null && values != null;

        double[][] verticalQueries = Matrix.transpose(queries);
        double[][] verticalKeys = Matrix.transpose(keys);
        double[][] verticalValues = Matrix.transpose(values);

        QKV.add(verticalQueries);
        QKV.add(verticalKeys);
        QKV.add(verticalValues);

        return QKV;
    }
}
