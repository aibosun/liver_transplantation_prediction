// get this webpage ready
$(document).ready(function() {
   //buildEchart();
   getFeatures();
});
function getFeatures(){
    $.ajax({
        type: "POST",
        url: "/getfeatures",
        // contentType: 'application/json;charset=UTF-8',
        // data: JSON.stringify(postdata),
        dataType: "json",
        success:function(result){
            //feturesdt = result["features"];
            var ydata =[];
            var dtvalue = [];
            for(var i=19;i>=0;i--){
                ydata[i]=result[i][0];
                dtvalue[i]=result[i][1];
            }
            buildEchartAutom(ydata,dtvalue);
        }

    });
}
// build the features bar chart
function buildEchartAutom(ydata,dtvalue){

   var myBChart = echarts.init(document.getElementById('featureBAR'));

   var content = ["ENCEPHALOPATHY AT TRANSPLANT"
   ,"DONOR SEROLOGY ANTI CMV-N"
   ,"DONOR SEROLOGY ANTI CMV-P"
   ,"DECEASED DONOR-THYROXINE-T4 B/N BRAIN DEATH W/IN 24 HRS OF PROCUREMENT"
   ,"DECEASED DONOR-NON-HEART BEATING DONOR"
   ,"CALCULATED RECIPIENT HEIGHT(cm)"
   ,"DONOR WEIGHT (KG)"
   ,"CALCULATED CANDIDATE BMI AT REMOVAL/CURRENT TIME"
   ,"CALCULATED CANDIDATE BMI AT REMOVAL/CURRENT TIME"
   ,"DECEASED DONOR-SYNTHETIC ANTI DIURETIC HORMONE (DDAVP)-Y"
   ,"CALCULATED DONOR HEIGHT (CM)"
   ,"DECEASED DONOR-SYNTHETIC ANTI DIURETIC HORMONE (DDAVP)-N",
   "RECIPIENT SERUM CREATININE AT TIME OF TX"
   ,"DAYS ON LIVER WAITING LIST"
   ,"DDR:Hematocrit:"
   ,"RECIPIENT SERUM ALBUMIN  @ TRANSPLANT"
   ,"DONOR AGE (YRS)"
   ,"THE NUMBER OF PREVIOUS TRANSPLANTS-Y"
   ,"THE NUMBER OF PREVIOUS TRANSPLANTS-N"
   ,"TRANSPLANT YEAR"];

   optionB = {
	   xAxis: {
		   type: 'log',
		   position:'top',
		   logBase:10
	   },
	   tooltip : {
		   trigger: 'axis',
		   axisPointer : {            // 坐标轴指示器，坐标轴触发有效
			   type : 'shadow'        // 默认为直线，可选为：'line' | 'shadow'
		   },
		   formatter: function (params) {
			   var indexnm = params[0].dataIndex;
			   var name =params[0].axisValue;
			   var contenty = content[indexnm];
			   var relVal = name+" : "+contenty+"<br/>";
			   return relVal;
		   }
	   },
	   yAxis: {
		   type: 'category',
		   data:ydata
	   },
	   series: [{
		   data:dtvalue,
		   type: 'bar',
		   barWidth: '25',
           itemStyle:{color:'#73a2ea'}
       }]
   };
   myBChart.setOption(optionB);
}

// build the features bar chart
function buildEchart(){

   var myBChart = echarts.init(document.getElementById('featureBAR'));
   var ydata = ['ENCEPH_TX'
   ,'CMV_DON_N'
   ,'CMV_DON_P'
   ,'PT_T4_DON_Y'
   ,'NON_HRT_DON_N'
   ,'HGT_CM_CALC'
   ,'WGT_KG_DON_CALC'
   ,'END_BMI_CALC'
   , 'BMI_CALC'
   ,'DDAVP_DON_Y'
   ,'HGT_CM_DON_CALC'
   ,'DDAVP_DON_N'
   ,'CREAT_TX'
   ,'DAYSWAIT_CHRON'
   , 'HEMATOCRIT_DON'
   , 'ALBUMIN_TX'
   , 'AGE_DON'
   , 'PREV_TX_Y'
   , 'PREV_TX_N'
   ,'TX_YEAR'];
   var content = ["ENCEPHALOPATHY AT TRANSPLANT"
   ,"DONOR SEROLOGY ANTI CMV-N"
   ,"DONOR SEROLOGY ANTI CMV-P"
   ,"DECEASED DONOR-THYROXINE-T4 B/N BRAIN DEATH W/IN 24 HRS OF PROCUREMENT"
   ,"DECEASED DONOR-NON-HEART BEATING DONOR"
   ,"CALCULATED RECIPIENT HEIGHT(cm)"
   ,"DONOR WEIGHT (KG)"
   ,"CALCULATED CANDIDATE BMI AT REMOVAL/CURRENT TIME"
   ,"CALCULATED CANDIDATE BMI AT REMOVAL/CURRENT TIME"
   ,"DECEASED DONOR-SYNTHETIC ANTI DIURETIC HORMONE (DDAVP)-Y"
   ,"CALCULATED DONOR HEIGHT (CM)"
   ,"DECEASED DONOR-SYNTHETIC ANTI DIURETIC HORMONE (DDAVP)-N",
   "RECIPIENT SERUM CREATININE AT TIME OF TX"
   ,"DAYS ON LIVER WAITING LIST"
   ,"DDR:Hematocrit:"
   ,"RECIPIENT SERUM ALBUMIN  @ TRANSPLANT"
   ,"DONOR AGE (YRS)"
   ,"THE NUMBER OF PREVIOUS TRANSPLANTS-Y"
   ,"THE NUMBER OF PREVIOUS TRANSPLANTS-N"
   ,"TRANSPLANT YEAR"];

   optionB = {
	   xAxis: {
		   type: 'log',
		   position:'top',
		   logBase:10
	   },
	   tooltip : {
		   trigger: 'axis',
		   axisPointer : {            // 坐标轴指示器，坐标轴触发有效
			   type : 'shadow'        // 默认为直线，可选为：'line' | 'shadow'
		   },
		   formatter: function (params) {
			   var indexnm = params[0].dataIndex;
			   var name =params[0].axisValue;
			   var contenty = content[indexnm];
			   var relVal = name+" : "+contenty+"<br/>";
			   return relVal;
		   }
	   },
	   yAxis: {
		   type: 'category',
		   data:ydata
	   },
	   series: [{
		   data: [
				 {value:4.46, itemStyle:{color:'#c4ccd3'}}
			   , {value:5.11, itemStyle:{color:'#40e0d0'}}
			   , {value:5.38, itemStyle:{color:'#dcc668'}}
			   , {value:5.57, itemStyle:{color:'#c23531'}}
			   , {value:5.65, itemStyle:{color:'#da70d6'}}
			   , {value:5.96, itemStyle:{color:'#314656'}}
			   , {value:13.53, itemStyle:{color:'#dd8668'}}
			   , {value:16.67, itemStyle:{color:'#91c7ae'}}
			   , {value:16.83, itemStyle:{color:'#314656'}}
			   , {value:16.95, itemStyle:{color:'#dd8668'}}
			   , {value:17.80, itemStyle:{color:'#91c7ae'}}
			   , {value:19.24, itemStyle:{color:'#c4ccd3'}}
			   , {value:20.46, itemStyle:{color:'#40e0d0'}}
			   , {value:33.37, itemStyle:{color:'#c23531'}}
			   , {value:46.07, itemStyle:{color:'#da70d6'}}
			   , {value:48.52, itemStyle:{color:'#314656'}}
			   , {value:68.96, itemStyle:{color:'#dcc668'}}
			   , {value:165.47, itemStyle:{color:'#91c7ae'}}
			   , {value:202.26, itemStyle:{color:'#a14858'}}
			   , {value:202.26, itemStyle:{color:'#dd8668'}}],
		   type: 'bar',
		   barWidth: '25'
	   }]
   };
   myBChart.setOption(optionB);
}
