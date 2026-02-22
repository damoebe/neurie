package me.damoebe.transformer.mha;

import me.damoebe.transformer.Embedding;

import java.util.ArrayList;
import java.util.List;

/**
 * The DecoderHead Object class
 */
public class DecoderHead extends Head{
    /**
     * The main constructor of the decoder Head
     * Identical to the super constructor
     * @param inputEmbeddingAmount The amount of input embeddings
     * @param inputEmbeddingSize   The size of each input embedding
     */
    protected DecoderHead(int inputEmbeddingAmount, int inputEmbeddingSize) {
        super(inputEmbeddingAmount, inputEmbeddingSize);
    }

    /**
     * Generates an output for the input, regarding the decoder head attention rules.
     * @param inputEmbeddings The input Embedding list
     * @return The output of this decoder head
     * @throws Exception see super class
     */
    @Override
    public List<Embedding> getOutputFor(List<Embedding> inputEmbeddings) throws Exception{

        if (!isValidInput(inputEmbeddings)) throw new Exception("Not a valid input Embedding list!");

        int weightIndex = 0;

        List<double[]> queries = new ArrayList<>();
        List<double[]> keys = new ArrayList<>();
        List<double[]> values = new ArrayList<>();

        for (Embedding inputEmbedding : inputEmbeddings){

            double[] query = new double[this.inputEmbeddingSize];
            double[] key = new double[this.inputEmbeddingSize];
            double[] value = new double[this.inputEmbeddingSize];

            int i = 0;
            for (Double embeddingValue : inputEmbedding.data()){
                query[i] = (embeddingValue * this.weights.get(weightIndex) + this.biases.get(weightIndex));
                weightIndex++;

                key[i] = (embeddingValue * this.weights.get(weightIndex) + this.biases.get(weightIndex));
                weightIndex++;

                value[i] = (embeddingValue * this.weights.get(weightIndex) + this.biases.get(weightIndex));
                weightIndex++;

                i++;
            }
            queries.add(query);
            keys.add(key);
            values.add(value);
        }

        double[][] scaled_masked_dot_product = new double[queries.size()][keys.size()];
        for (int q = 0; q != queries.size(); q++){
            for (int k = 0; k != keys.size(); k++){
                // triangular mask (causal mask)
                if (k > q){
                    scaled_masked_dot_product[q][k] = -Double.MAX_VALUE;
                }

                // calculate dot product (sum(vec*vec))
                double sum = 0;
                double[] query = queries.get(q);
                double[] key = keys.get(k);
                for (int i = 0; i != query.length; i++){
                    sum *= query[i] * key[i];
                }
                scaled_masked_dot_product[q][k] = sum / Math.sqrt(key.length); // scaling
            }
        }

        //TODO: apply softmax and multiply with values


        return null;
    }

}
