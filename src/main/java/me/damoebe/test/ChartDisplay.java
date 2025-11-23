package me.damoebe.test;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.DefaultXYDataset;

import javax.swing.*;
import java.awt.*;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;


public class ChartDisplay {

    private JFrame frame;
    private Map<String, double[][]> dataArrayMap = new HashMap<>();
    private DefaultXYDataset dataset;
    private int maxArraySize;
    private JFreeChart chart;

    private final String name;
    private final String xName;
    private final String yName;


    public ChartDisplay(String name, String xName, String yName, int maxArraySize){
        frame = new JFrame();
        frame.setTitle("Chart");
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);

        dataset = new DefaultXYDataset();

        this.name = name;
        this.xName = xName;
        this.yName = yName;
        this.maxArraySize = maxArraySize;

        chart = ChartFactory.createXYLineChart(name, xName, yName, dataset);
        ChartPanel chartPanel = new ChartPanel(chart);
        frame.setContentPane(chartPanel);

        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setExtendedState(frame.getExtendedState() | JFrame.MAXIMIZED_BOTH);
        frame.setVisible(true);
    }

    public void update(double data1, double data2, String series){

        dataArrayMap.computeIfAbsent(series, k -> new double[2][]);

        // setting up new data size
        int newDataSize;
        if (dataArrayMap.get(series)[0] == null){
            newDataSize = 1;
        }else {
            // check dataSize
            if ((dataArrayMap.get(series)[0].length > maxArraySize) && (maxArraySize != 0)){
                double[][] newMap = dataArrayMap.get(series);
                newMap[0] = Arrays.copyOfRange(newMap[0], 1, newMap[0].length);
                newMap[1] = Arrays.copyOfRange(newMap[1], 1, newMap[1].length);
                dataArrayMap.put(series, newMap);
            }

            newDataSize = dataArrayMap.get(series)[0].length + 1;
        }

        double[][] newDataArray = new double[2][newDataSize];

        // clone dataArray if needed
        if (dataArrayMap.get(series)[0] != null) {
            for (int i = 0; i != newDataSize - 1; i++) {
                newDataArray[0][i] = dataArrayMap.get(series)[0][i];
                newDataArray[1][i] = dataArrayMap.get(series)[1][i];
            }

        }

        // setting new data
        newDataArray[0][newDataSize-1] = data1;
        newDataArray[1][newDataSize-1] = data2;

        dataArrayMap.put(series, newDataArray);
        dataset.addSeries(series, dataArrayMap.get(series));

        // update frame
        chart = ChartFactory.createXYLineChart(name, xName, yName, dataset);
        ChartPanel chartPanel = new ChartPanel(chart);

        for (int i = 0; i != chart.getXYPlot().getSeriesCount(); i++) {
            chartPanel.getChart().getXYPlot().getRenderer().setSeriesStroke(i, new BasicStroke(3.0f));
        }

        frame.setContentPane(chartPanel);
        frame.revalidate();
        frame.repaint();
    }

}
