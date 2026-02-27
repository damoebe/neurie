package me.damoebe.transformer.embedding;

import java.util.ArrayList;
import java.util.List;

/**
 * Record class for Transformer intern data.
 * @param data The data this Embedding contains as a Double List.
 */
public record Embedding(List<Double> data) {

    /**
     * Constructs an empty Embedding-Object
     */
    public Embedding(){
        this(new ArrayList<>());
    }

    /**
     * Loads an Embedding object data from a token by positional encoding it.
     * @param token The token that should be transformed into an embedding.
     * @param tokenPos The position of the provided Token object
     * @return The updated object from which the function is called
     */
    public Embedding fromToken(Token token, int tokenPos){
        List<Double> posEncodings = new ArrayList<>();
        int i = 0;
        for (Double tokenValue : token.data()){
            if (i % 2 == 0){
                posEncodings.add(Math.sin((double) tokenPos /10000*((double) (2 * i) /token.data().size())));
            }else{
                posEncodings.add(Math.cos((double) tokenPos /10000*((double) (2 * i) /token.data().size())));
            }
            i++;
        }
        data.clear();
        for (int dataIndex = 0; dataIndex != token.data().size(); dataIndex++){
            data.add(posEncodings.get(dataIndex) + token.data().get(dataIndex));
        }
        return this;
    }

    /**
     * Getter for the size of the data array.
     * @return The data Double List size.
     */
    public int getEmbeddingSize(){
        return data.size();
    }
}
