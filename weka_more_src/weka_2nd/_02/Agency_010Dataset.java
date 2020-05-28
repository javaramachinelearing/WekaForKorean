package weka_2nd._02;

import java.io.File;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka_2nd.MoreWekaCommon;

public class Agency_010Dataset {
	
	private Instances data = null;
	private Instances data_sel_attr = null;
	private Agency_031FeatureSel_Result[] result = null;
	
	public static void  main (String args[]) throws Exception{
		Agency_010Dataset obj = new Agency_010Dataset();
		obj.openArffonWeb();
		obj.preprocess();
		obj.feature();
		obj.decisionMaking();
	}

	// 1) data loader
	public void openArffonWeb() throws Exception{    
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File(MoreWekaCommon.arffPath+"agency\\speed.csv"));	
		this.data = loader.getDataSet();
		System.out.println("��ü ������ �� �� : " + data.size());
	}
	
	// 2) ��ó�� (���� ���͸�)
	public void preprocess() throws Exception{
		Agency_020Preprocess preprocess 
		         = new Agency_020Preprocess(this.data);
		
		this.data_sel_attr = preprocess.action();		
	}

	// 3) Ư¡(�Ӽ�) ����
	public void feature() throws Exception{
		Agency_030SelectFeature feature = new Agency_030SelectFeature();
		this.result = feature.attrSelFeature(this.data_sel_attr);
	}
	
	// 4) �� �з�
	public void decisionMaking() throws Exception{
		Agency_04DecisionMaking decision 
        = new Agency_04DecisionMaking("�ý��� ���� �������� ����� �׷���", this.result);
		
		decision.radarChart();
		
	}
	
}
