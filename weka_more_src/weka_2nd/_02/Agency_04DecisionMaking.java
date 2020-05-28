package weka_2nd._02;

import java.awt.Font;

import javax.swing.JFrame;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.labels.StandardCategoryToolTipGenerator;
import org.jfree.chart.plot.SpiderWebPlot;
import org.jfree.chart.title.LegendTitle;
import org.jfree.chart.title.TextTitle;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.ui.RectangleEdge;
import org.jfree.ui.RefineryUtilities;

import weka_2nd.MoreWekaCommon;
/***
 * https://www.boraji.com/jfreechart-polar-chart-example
 * jfreechart-1.0.19.jar , jcommon-1.0.23.jar
 * 
 * */
@SuppressWarnings("serial")
public class Agency_04DecisionMaking extends JFrame {


	   public Agency_04DecisionMaking(String title, Agency_031FeatureSel_Result[] result) {
		     super(title);
	         CategoryDataset dataset = createDataset(result);
	         JFreeChart chart = createChart(dataset);
	         ChartPanel chartPanel = new ChartPanel(chart);
	         chartPanel.setPreferredSize(new java.awt.Dimension(900, 1500));
	         setContentPane(chartPanel);	      
	   }
	   

	   private static CategoryDataset createDataset(Agency_031FeatureSel_Result[] result) {
		   		 
		    //������ ������ ����
		    DefaultCategoryDataset dataset = new DefaultCategoryDataset();
		    dataset.addValue(0,"����","ȭ���");
		    dataset.addValue(0,"����","���뼱 �ӵ�");
		    dataset.addValue(0,"����","platform");
		    dataset.addValue(0,"����","����");
		 
		    for (int x=0 ; x < result.length ; x++){
		    	for (int y=0 ; y < result[x].weight.length ; y++){
		    		
//		    		System.out.println("weight (value) : " + result[x].weight[y] + 
//		    				           " , attr. algo. (series) " + MoreWekaCommon.getFeatureAlgorithmName(result[x].attrEval) +
//		    				           " , attr. name (category) : " + result[x].attrName[y]
//		    				          );
		    		if(result[x].attrName[y] != null)
			    		dataset.addValue(result[x].weight[y], 
			    				         MoreWekaCommon.getFeatureAlgorithmName(result[x].attrEval), 
			    				         result[x].attrName[y]);
		    	}	
		    }
		    
		    return dataset;
		 
		}
	   	
	   private static JFreeChart createChart(CategoryDataset dataset) {
		    SpiderWebPlot plot = new SpiderWebPlot(dataset);
		    plot.setLabelFont(new Font("�������",0,20));
		    plot.setStartAngle(0);
		    plot.setInteriorGap(0.40);
		    plot.setToolTipGenerator(new StandardCategoryToolTipGenerator());
		    JFreeChart chart = new JFreeChart("�ý��� ���� �������� ����� �׷���",
		    TextTitle.DEFAULT_FONT, plot, false);
		    LegendTitle legend = new LegendTitle(plot);
		    legend.setPosition(RectangleEdge.BOTTOM);
		    chart.addSubtitle(legend);  
		    return chart;  
		}
	   
	   
	   public void radarChart() throws Exception{
		    pack();
		    RefineryUtilities.centerFrameOnScreen(this);
		    setVisible(true);		   
	   }
	   
}
