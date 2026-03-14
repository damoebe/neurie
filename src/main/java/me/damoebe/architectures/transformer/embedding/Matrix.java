package me.damoebe.architectures.transformer.embedding;


public class Matrix {

    /**
     * Multiplies two matrices with each other
     * @param matrix1 The first matrix
     * @param matrix2 The second matrix
     * @return The result matrix of the multiplication
     */
    public static double[][] multiply(double[][] matrix1, double[][] matrix2){
        if (matrix1[0].length != matrix2.length) return null;

        double[][] resultMatrix = new double[matrix1.length][matrix2[0].length];

        for (int row = 0; row < matrix1.length; row++) {
            for (int col = 0; col < matrix2[0].length; col++) {
                double sum = 0;
                for (int k = 0; k < matrix1[0].length; k++) {
                    sum += matrix1[row][k] * matrix2[k][col];
                }
                resultMatrix[row][col] = sum;
            }
        }
        return resultMatrix;
    }

    /**
     * Adds two matrices of same dimension
     * @param matrix1 The first matrix
     * @param matrix2 The second matrix
     * @return The sum matrix
     */
    public static double[][] add(double[][] matrix1, double[][] matrix2){
        if (matrix1.length != matrix2.length || matrix1[0].length != matrix2[0].length) return null;

        double[][] resultMatrix = new double[matrix1.length][matrix2[0].length];
        for (int row  = 0; row != matrix2.length; row++){
            for (int colum = 0; colum != matrix2[0].length; colum++){
                resultMatrix[row][colum] = matrix2[row][colum] + matrix1[row][colum];
            }
        }
        return resultMatrix;
    }

    /**
     * Transposes a matrix
     * @param matrix The matrix that should be transposed
     * @return The transposed matrix (this.array2d)
     */
    public static double[][] transpose(double[][] matrix){
        double[][] resultMatrix = new double[matrix[0].length][matrix.length];
        for (int row = 0; row != matrix.length; row++){
            for (int col = 0; col != matrix[0].length; col++){
                resultMatrix[col][row] = matrix[row][col];
            }
        }
        return resultMatrix;
    }
}
