package me.damoebe.transformer.mha;

import me.damoebe.transformer.embedding.Embedding;
import me.damoebe.transformer.embedding.Sequence;

import java.util.ArrayList;
import java.util.List;

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
    protected List<List<double[]>> getQKVMatrices(Sequence[] inputEmbeddings) throws Exception{
        if (inputEmbeddings.length != 2) throw new Exception("The EDHead class has to be provided with exactly 2 inputs!");
        List<List<double[]>> QKV = new ArrayList<>();
        int weightIndex = 0;

        List<double[]> queries = new ArrayList<>();
        List<double[]> keys = new ArrayList<>();
        List<double[]> values = new ArrayList<>();

        for (Embedding decoderEmbedding : inputEmbeddings[0].embeddings()){
            double[] query = new double[this.inputEmbeddingSize];
            int i = 0;
            for (Double embeddingValue : decoderEmbedding.data()){
                query[i] = (embeddingValue * this.weights.get(weightIndex) + this.biases.get(weightIndex));
                weightIndex++;
                i++;
            }
            queries.add(query);
        }

        for (Embedding encoderEmbedding : inputEmbeddings[1].embeddings()){
            double[] key = new double[this.inputEmbeddingSize];
            double[] value = new double[this.inputEmbeddingSize];
            int i = 0;
            for (Double embeddingValue : encoderEmbedding.data()){
                key[i] = (embeddingValue * this.weights.get(weightIndex) + this.biases.get(weightIndex));
                weightIndex++;

                value[i] = (embeddingValue * this.weights.get(weightIndex) + this.biases.get(weightIndex));
                weightIndex++;

                i++;
            }
            keys.add(key);
            values.add(value);
        }

        QKV.add(queries);
        QKV.add(keys);
        QKV.add(values);

        return QKV;
    }
}
