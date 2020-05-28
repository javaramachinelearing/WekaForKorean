package weka_2nd._05.bird;


import weka.classifiers.Classifier;
import weka.core.Instances;
import weka_2nd.MoreWekaCommon;

public class Image_040Classify {

	private Classifier model = null;
		
	public Image_040Classify (Classifier model){
		this.model = model;
	}
	
	public Object loadModel(String modelName) throws Exception{
		return weka.core.SerializationHelper.read(MoreWekaCommon.modelPath +"image\\image_bird_" 
				                                  +modelName);
	}
	
	public void useModel (Instances test) throws Exception{
			
		Classifier model = (Classifier)loadModel(MoreWekaCommon.getModelName(this.model));
		System.out.println("Classifier model : " + MoreWekaCommon.getModelName(model));
		test.setClassIndex(test.numAttributes()-1);
		
		System.out.println(test.relationName());
		
//		model.buildClassifier(test);
		
		// ���� �űԵ����� (�ܾ�)�� ������ ��ġ���� �Ǵ�
		int inCorrectCnt = 0;
		for(int x=0 ; x < test.numInstances(); x++){	
			
			if(test.get(x).classValue() != model.classifyInstance(test.get(x))){
				inCorrectCnt ++;
				System.out.println(
						(x+1) + " st : " +
						" test : model = " + 
						test.get(x).classValue() + 
						" : " + 
						model.classifyInstance(test.get(x)) +
						(test.get(x).classValue() == model.classifyInstance(test.get(x))? " ��ġ" : " ����ġ")
						+ " : " +test.get(x)
						);			
			}
		}	
		System.out.println("����ġ �Ǽ� : " + inCorrectCnt + 
				          " , ����ġ�� : " + String.format("%.1f", (double)inCorrectCnt/(double)test.numInstances() * 100) + " %");
	}
}
