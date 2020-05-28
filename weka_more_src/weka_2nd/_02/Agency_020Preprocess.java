package weka_2nd._02;

import java.awt.Dimension;

import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import weka.core.Instance;
import weka.core.Instances;

public class Agency_020Preprocess {

	private Instances data = null;
	private Instances data_melt = null;
	private Instances data_sel_attr = null;
	
	private Agency_021melt plugin = null;
	private Agency_022Filter filter = null;

	public Agency_020Preprocess (Instances data){
		this.data = data;
	}
	
	public Instances action() throws Exception{
		this.showRules("������ ������",data);

		this.plugin = new Agency_021melt(this.data);
		this.data_melt = plugin.melt("1-4");
		this.showRules("������ ������", data_melt);

		this.filter = new Agency_022Filter(this.data_melt);
		this.data_sel_attr = filter.filter(); 
		this.showRules("���͸� �� ������ ������", data_sel_attr);
		
		return this.data_sel_attr;
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
	 
}
