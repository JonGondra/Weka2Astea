package ehu.weka;

import weka.core.Instances;

public class Main {
    public static void main(String[] args) throws Exception {
        Weka weka = Weka.getInstance();

        String path = "C:\\Users\\jongo\\OneDrive\\Escritorio\\WEKA\\1.PRAKTIKA\\DATUAK 1.PRAKTIKA\\heart-c.arff";
        Instances data = weka.fitxategiaKargatu(path);

        weka.atributuKop(data);
        weka.instantziaKop(data);

        String[] atributuLista = weka.atributuLista(data);
        for(int i=0; i<atributuLista.length; i++){
            int zenb = weka.missingKop(data,atributuLista[i]);
        }

        weka.klaseMinoritarioa(data);

        weka.kfCV(data, 3);

        weka.holdOutEgin(path);


    }
}
