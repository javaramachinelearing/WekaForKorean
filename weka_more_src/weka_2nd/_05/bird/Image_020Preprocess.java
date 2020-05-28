package weka_2nd._05.bird;

import java.awt.Dimension;

import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.JTable;

import weka.core.Instance;
import weka.core.Instances;

public class Image_020Preprocess {

	private Instances data = null;
	
	public Image_020Preprocess (Instances data) throws Exception{
		this.setData(data);
	}
	
	public Instances action() throws Exception{

		this.showRules("orginal data ", this.getData());
		Instances filteredData = this.setImageFilter(this.getData());
		this.showRules("filtered data ��ġȭ �Ӽ����� : " + filteredData.numAttributes(), filteredData);
		
		return filteredData;
	}
	
	public Instances setImageFilter (Instances image) throws Exception{
		Image_021Filter filter =  new Image_021Filter(image);
		Instances filteredImage = filter.setImageFilters()
				                    .getData();
		filteredImage.setClassIndex(0); // class = first attr
		return filteredImage;
	}
	
	// swing ���̺� ���� ����
	 public void showRules(String title, Instances filteredData) throws Exception{

		 // �Ӽ��� ���� (header)
		 String header[] = new String[filteredData.numAttributes()];

		 for (int i = 0; i < filteredData.numAttributes() ; i++) 
			header[i] = filteredData.attribute(i).name();
		 
		 // ������ �� ���� (contents)
		 String contents[][] = new String[filteredData.numInstances()][filteredData.numAttributes()];

		 for(int x=0 ; x < filteredData.numInstances(); x++){
			 Instance row = filteredData.get(x);
			 for (int y = 0; y < row.numAttributes(); y++) {
				try{
					contents[x][y] = row.stringValue(y);
				}catch (IllegalArgumentException iae){
					contents[x][y] = row.value(y)+"";
				}	
			}
		}
		this.makeTable(title, header, contents);
	 }
	 
	 // swing ���̺�� ��� ���
	 public void makeTable(String title, String[] header, String[][] contents){
		 Dimension dim = new Dimension(1024,768);
		 JFrame frame = new JFrame(title);
		 frame.setLocation(100, 200);
		 frame.setPreferredSize(dim);
		 
		 JTable table = new JTable(contents, header);
		 JScrollPane scrollpane = new JScrollPane(table);
		
		 frame.add(scrollpane);
		 frame.pack();
		 frame.setVisible(true);
	 }

	public Instances getData() {
		return data;
	}

	public void setData(Instances data) {
		this.data = data;
	}
}
