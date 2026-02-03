[![](https://jitpack.io/v/damoebe/neurie.svg)](https://jitpack.io/#damoebe/neurie) ![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/damoebe/neurie)
# neurie API
Neurie is a Java framework, which offers you simple and easy to understand neural network architechtures to use in your project. You can train the networks by feeding them dataset in a .json format (see below). Neurie can be trained with two learning techniques: Standart Deep Learning (using Backpropagation) and Evolution-Learning. You can adjust the size of the network, as well as the learing rate and background noise. Moreover neurie network performances can be visually displayed by using the build-in chart-generator class to test archiechure performances. You can also view the documentation of each class used for neurie to understand the math behind the networks and take full advantage of the flexibility of neurie networks. The Evolution Learing is based on an epoch system, which uses target values from the dataset to calculate the networks fittness factor and the deep learning network uses the sigmoid function as an activation function to update neuron activations. Neurie is (for now) only available as a feed-forward architechture (Not Recurrent), which might change in the future. 
## How To Use The API?
Neurie is a Java Maven project, so you will only be able to use neurie in Maven or Gradle. You will need to use the jitpack repository to access the API, because neurie has not been uploaded to maven central. If you are using Maven, paste this repository into your repositories section in the pom.xml:
```
<repository>
    <id>jitpack.io</id>
    <url>https://jitpack.io</url>
</repository>
```
and paste this dependency into your dependencies section:
```
<dependency>
    <groupId>com.github.damoebe</groupId>
    <artifactId>neurie</artifactId>
    <version>v1.1</version>
</dependency>
```
In case you are using Gradle, you will need to paste `maven { url 'https://jitpack.io' }` in in your repositories section in your settings.gradle and `implementation 'com.github.damoebe:neurie:v1.1'` in your dependencies section.\
Reload Maven/Gradle and you will be setup and can now access the Java classes of neurie. Make sure to download the Java documentation to discover the meaning of methods, classes and attributes.

