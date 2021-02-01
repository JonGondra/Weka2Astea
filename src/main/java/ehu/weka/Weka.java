package ehu.weka;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.util.Random;

public class Weka {

    private static final Weka instance = new Weka();

    public static Weka getInstance() {
        return instance;
    }

    private Weka() { }

    public Instances fitxategiaKargatu(String fitx) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(fitx);
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        return data;
    }

    public int atributuKop(Instances data){

        System.out.println("Datu sorta honetan " + data.numAttributes() + " atributu daude.\n");
        return data.numAttributes();

    }

    public int instantziaKop(Instances data){

        System.out.println("Datu sorta honetan " + data.numInstances() + " atributu daude.\n");
        return data.numInstances();

    }

    public int missingKop(Instances data, String atrib){

        Attribute attribute = data.attribute(atrib);
        System.out.println("Datu sorta honetan,"+atrib+" atributuak  " + data.attributeStats(attribute.index()).missingCount + " missing values ditu.\n");
        return data.attributeStats(attribute.index()).missingCount;

    }

    public String[] atributuLista(Instances data) throws Exception {

        String[] emaitza = new String[data.numAttributes()];

        for(int i=0; i<emaitza.length; i++){
            emaitza[i] = data.attribute(i).name();
        }

        return emaitza;
    }

    public void klaseMinoritarioa(Instances data) throws Exception {

        Attribute klasea = data.classAttribute();

        String izena = klasea.value(0);
        int[] kop = data.attributeStats(data.numAttributes()-1).nominalCounts;
        int min = kop[0];

        for(int i=0; i<kop.length; i++) {
            System.out.println(klasea.value(i)+": "+kop[i]);
            if(kop[i]<min && kop[i]!=0){
                min = kop[i];
                izena = klasea.value(i);
            }
        }
        System.out.println("Klase minoritarioa "+izena+ " da : "+min+"\n");
    }

    private NaiveBayes naiveBayes(Instances data) throws Exception {

        NaiveBayes model=new NaiveBayes();
        //ez dugu entrenatu behar
        //model.buildClassifier(data);

        return model;
    }

    public void kfCV(Instances data, int k) throws Exception {

        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(naiveBayes(data), data, k, new Random(1));

        //Matrizea inprimatu
        System.out.println(eval.toMatrixString());
        precision(eval,data.classAttribute());
        weightedAvg(eval);

    }

    private Double[] precision(Evaluation eval, Attribute klasea){
        Double[] precisiones = new Double[klasea.numValues()];
        System.out.println("Class \t Precision");
        for(int i=0; i<klasea.numValues();i++){
            precisiones[i]=eval.precision(i);
            System.out.println(klasea.value(i)+" \t"+ precisiones[i]);
        }
        return precisiones;
    }


    private Double[] weightedAvg(Evaluation eval){
        Double[] weightedAvg = new Double[8];
        weightedAvg[0]=eval.weightedTruePositiveRate();
        weightedAvg[1]=eval.weightedFalsePositiveRate();
        weightedAvg[2]=eval.weightedPrecision();
        weightedAvg[3]=eval.weightedRecall();
        weightedAvg[4]=eval.weightedFMeasure();
        weightedAvg[5]=eval.weightedMatthewsCorrelation();
        weightedAvg[6]=eval.weightedAreaUnderROC();
        weightedAvg[7]=eval.weightedAreaUnderPRC();

        System.out.println("\nTP Rate \t \t \t FP Rate \t \t \t \t Precision \t \t \t \t Recall \t \t \t \t F-Measure \t \t \t \t MCC \t \t \t \t \t ROC Area \t \t \t \t PRC Area");
        for(int i=0; i<weightedAvg.length;i++){
            System.out.print(weightedAvg[i]+" \t ");
        }
        return weightedAvg;
    }

    public void holdOutEgin(String path) throws Exception {
        //Datuak prestatu
        //Randomize data
        Instances data = fitxategiaKargatu(path);
        Randomize filter = new Randomize();
        filter.setInputFormat(data);
        Instances RandomData = Filter.useFilter(data,filter);

        //Split Data
        SplitData(RandomData);
    }


    private void SplitData(Instances RandomData) throws Exception{
        //Split the data
        RemovePercentage filterRemoveTest = new RemovePercentage();
        filterRemoveTest.setInputFormat(RandomData);

        //%30 hartu (test)
        filterRemoveTest.setPercentage(70);
        Instances test = Filter.useFilter(RandomData,filterRemoveTest);
        test.setClassIndex(test.numAttributes() - 1);
        System.out.println("\n\nTest instantzia kopurua: "+test.numInstances());

        //%70 hartu (train)
        RemovePercentage filterRemoveTrain = new RemovePercentage();
        filterRemoveTrain.setInputFormat(RandomData);
        filterRemoveTrain.setPercentage(70);
        filterRemoveTrain.setInvertSelection(true);
        Instances train = Filter.useFilter(RandomData,filterRemoveTrain);
        train.setClassIndex(train.numAttributes() - 1);
        System.out.println("Train instantzia kopurua: "+train.numInstances()+"\n");

        HoldOut(train,test);
    }


    private Evaluation HoldOut(Instances train, Instances test) throws Exception {
        NaiveBayes model = new NaiveBayes(); //Build model
        model.buildClassifier(train); //Hold-out denez, entrenatu behar da

        //Ebaluatu
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(model,test);
        eval.toMatrixString();

        System.out.println("Estimated Accuracy: " + Double.toString(eval.pctCorrect()));
        System.out.println("Confusion Matrix: " + eval.toMatrixString("num"));

        return eval;
    }


}
