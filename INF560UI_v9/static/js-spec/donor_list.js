//get ready for the global variables to store data from uploaded CSV files
var recipientdata=[];
var recipInfoPass=[];
var dondata=[];
var matchscore;
var predictscore;
var meldrange;
// get ready for this webpage
$(document).ready(function() {
    $("#pid").hide();
    $("#selectbox_id").bind('change', function(){
        filterRecipient(this.value);
    });
});
// click "submit" button for uploading recipeint CSV file
function uploadRecip(){
    var selectedFile = document.getElementById('fileRecipid').files[0];
    if(selectedFile===undefined){
        alert("Please select recipient file!");
    }else {
        Papa.parse(selectedFile, {
           complete: function(results) {
               $("#pid").show();
               recipientdata = results.data;
               filterRecipient(200);
          }
       });
    }

}
// filter the uploaded recipient information according selected Meldscore range
function filterRecipient(meldrange){
    for(var i=1;i<=dondata.length-1;i++){
        $("#res"+i).hide();
    }
    $("#aboDIV").hide();
    $('#resultsid').hide();
    $("#resultid").hide();
    var rowArr = [];
    recipInfoPass = [];
    recipInfoPass[0] = recipientdata[0];
    var indexMeld = getCol("FINAL_MELD_PELD_LAB_SCORE","r");
    var j = 1;
    if(meldrange==200){
        recipInfoPass = recipientdata;
    }else if(meldrange==101){
        for(var i=1;i<=recipientdata.length-1;i++){
            if(parseInt(recipientdata[i][indexMeld])>101){
                rowArr = recipientdata[i];
                recipInfoPass[j]=rowArr;
                j = j+1;
            }
        }
    }else{
        for(var i=1;i<=recipientdata.length-1;i++){
            if(parseInt(recipientdata[i][indexMeld])>(meldrange-20) && parseInt(recipientdata[i][indexMeld])<=meldrange){
                rowArr = recipientdata[i];
                recipInfoPass[j]=rowArr;
                j = j+1;
            }
        }
    }
    buildRecipTable(recipInfoPass);
}
// (returen index by name) According columnname in CSV File return the index of that column in dataset  eg) FINAL_MELD_PELD_LAB_SCORE  reuturn index=4
function getCol(name,flagArr){
    var columnname =[];
    if(flagArr==="r"){
        columnname = recipientdata[0];
    }else{
        columnname = dondata[0];
    }

    for(var i=0;i<=columnname.length-1;i++){
        if(columnname[i]===name){
            return i;
        }
    }
}
// build recipient table on webpage
function buildRecipTable(infodata){
    var indexMeld = getCol("FINAL_MELD_PELD_LAB_SCORE","r");
    var indexAge = getCol("AGE","r");
    $('#recipienttable').html("");
    var htmlstr = "";
    htmlstr=htmlstr+'<tr>'
       +' <th class="w150">RECIPIENT_ID</th>'
       + ' <th class="w60">AGE</th>'
       +'  <th class="w60">GENDER</th>'
       +'  <th class="w60">ABO</th>'
       +'  <th class="w60">FINAL_MELD</th>'
       +'</tr>';

    for(var i=1;i<=infodata.length-1;i++){
      htmlstr=htmlstr+ '<tr><td class="w150">'+infodata[i][0]+'</td>'
               +'<td class="w60">'+infodata[i][indexAge]+'</td>'
               +'<td class="w60">'+infodata[i][2]+'</td>'
               +'+<td class="w60">'+infodata[i][3]+'</td>'
               +'+<td class="w60">'+infodata[i][indexMeld]+'</td></tr>';
   }
   $('#recipienttable').html(htmlstr);
   $("#id1").show();
   $("#id2").show();
   $("#donorSearchid").show();
   $("#donnotice").show();
}
// click "submit" button for uploading Donor CSV file
function upload(){
      var selectedFile = document.getElementById('fileid').files[0];
      if(selectedFile===undefined){
          alert("Please select donor file!");
      }else {
          Papa.parse(selectedFile, {
    	    complete: function(results) {
               dondata = results.data;
               var indexABO = getCol("ABO_DON","d");
               var indexGender = getCol("GENDER_DON","d");
               var htmlstr = "";
               htmlstr=htmlstr+'<tr>'
                  +' <th class="w150">DONOR_ID</th>'
                  + ' <th class="w70">BMI</th>'
                  +'  <th class="w60">ABO_DON</th>'
                  +'  <th class="w60">GENDER_DON</th>'
                  +' <th class="w60">RESULT</th>'
                  +'</tr>';

               for(var i=1;i<=dondata.length-1;i++){
                 htmlstr=htmlstr+ '<tr><td class="w150">'+dondata[i][0]+'</td>'
                          +'<td class="w70">'+dondata[i][1]+'</td>'
                          +'<td class="w60">'+dondata[i][indexABO]+'</td>'
                          +'+<td class="w60">'+dondata[i][indexGender]+'</td>'
                         +'<td class="w60"><a id="res'+i+'" href="javascript:void();" onclick="buildRtable('+i+','+dondata[i][0]+')" class="btn1 f16 fb">check</a></td></tr>';
              }
              $('#donortable').html(htmlstr);
              for(var i=1;i<=dondata.length-1;i++){
                  $("#res"+i).hide();
              }
               $("#tableid").show();
               $("#id11").show();


            }
         });
      }
}
// click "match" button to call prediction and matching function in python file
function doMatch(){
    $("#aboDIV").hide();
    $('#resultsable').html("");
    var allrecip = {"allrecip":recipientdata};
    var don = {"donor":dondata};
    var meld = $("#selectbox_id").val();
    var meldrange = {"meldrange":meld};
    for(var i=1;i<=dondata.length-1;i++){
        $("#res"+i).hide();
    }
    var postdata = $.extend({}, meldrange, don,allrecip);
    if(recipInfoPass.length!=1&&(don.length!=1||don.length!=0)){
        $("#spinner").show();
        $.ajax({
            type: "POST",
            url: "/doprediction",
            contentType: 'application/json;charset=UTF-8',
            data: JSON.stringify(postdata),
            dataType: "json",
            success:function(result){
                matchscore = result["match"];
                predictscore=result["predict"];
                for(var i=1;i<=dondata.length-1;i++){
                    $("#res"+i).show();
                }
                 $("#spinner").hide();
            }

        });
    }else{
        alert("No Recipient or Donor Infomation! ");
    }
}
// build results table after sucssfully calling doMatch() fucntion
function buildRtable(num,donorid){
    $('#resultsid').show();
    // num from  1
    $('#resultsable').html("");
    var indexnum = num-1;
    var donor_res = matchscore[indexnum];
    var htmlstrr = "";
    var datapre = [];
    htmlstrr=htmlstrr+'<tr>'
       +'<th class="w60">RECIPIENTID of DONOR: '+donorid+'</th>'
       +'<th class="w60">MATCHINGSCORE</th>'
       +'<th  class="w60">PREDICTIONSCORE</th>'
       +'<th  class="w60">MELDSCORE</th>'
       +'</tr>';

    for(var i=0;i<donor_res.length;i++){
      datapre[i]=[donor_res[i][0]+"",donor_res[i][1]*5,0];
      htmlstrr=htmlstrr+ '<tr><td class="w60">'+donor_res[i][0]+'</td>'
               +'<td class="w60">'+donor_res[i][1]+'</td>';
        for(var j=0 ; j<predictscore.length;j++){
            if(donor_res[i][0]==predictscore[j][0]){
                    datapre[i][2]=predictscore[j][1]*10;
                   htmlstrr=htmlstrr+'<td class="w60">'+predictscore[j][1]+'</td>';
            }
        }
        // -----------MELDDDDDD
        var indexMeld = getCol("FINAL_MELD_PELD_LAB_SCORE","r");
        for(var p =1;p<recipInfoPass.length;p++){
            if(donor_res[i][0]==recipInfoPass[p][0]){
                    var meldscoreR=recipInfoPass[p][indexMeld];
                   htmlstrr=htmlstrr+'<td class="w60">'+meldscoreR+'</td></tr>';
            }
        }

    }
    $('#resultsable').html(htmlstrr);
    $("#resultid").show();
    $("#resnotice").show();
    $("#aboDIV").show();

    preEchart(datapre,donorid);
}
// get 'color, data , presenting format' ready for the graph on the result part
function preEchart(datapre,donorid){
    // color get
    var colornodes =['#dd8668','#da70d6','#dcc668','#40e0d0','#a14858'];
    var dataE =[];
    var linkE =[];
    donorid = donorid+"";
    dataE[0]={name: donorid, itemStyle: {normal: {color: '#6fa1f2'}}, symbolSize: 80};
    for(var i=0;i<=datapre.length-1;i++){
        dataE[i+1]={name: datapre[i][0],value:datapre[i][2], itemStyle: {normal: {color: colornodes[i]}}, symbolSize: datapre[i][1]};
        linkE[i] = {source: donorid, target: datapre[i][0], lineStyle:{width:datapre[i][2]}};
    }
    buildEchart(dataE,linkE);
}
// builid the result graph on webpage of the result part
function buildEchart(dataE,linkE){
    $("#aboDIV").show();
    var eContainer = document.getElementById('aboDIV');
    var myChart = echarts.init(eContainer);

    option = {
        title: {},
        tooltip : {
             enterable:true,
             trigger: 'item',
             axisPointer : {
                    type : 'shadow'
             },
             formatter: function (params) {
                 var name = params.data.name;
                 var predictionscore = Math.round(params.data.value/10* 1000) / 1000;
                 var matchingscore = Math.round(params.data.symbolSize/5* 1000) / 1000;
                 var relVal ="";

                 if( params.data.itemStyle === undefined){
                     relVal=="";
                 }else{
                     if(matchingscore!=16){
                         relVal += "Recipient : "+name+"<br/>";
                         relVal +=' MatchingScore : ' + matchingscore+"<br/>";
                         relVal +='PredictionScore : ' +predictionscore+"<br/>";
                     }else{
                         relVal += "Donor : "+name+"<br/>";
                     }
                }
                 return relVal;
             }
        },
        animationDuration: 1000,
        animationEasingUpdate: 'quinticInOut',
        series : [
            {
                roam: false,
                type: 'graph',
                layout: 'force',
                label: {normal: {show: true}},
                data:dataE,
                links:linkE,
                edgeSymbol: ['', 'arrow'],
                force: {
                    repulsion: 8000,
                    edgeLength: 0.2
                },
                lineStyle: {
                    normal: {
                        opacity: 0.9,
                        color: 'target',
                        curveness: 0.0
                    }
                },

            }
        ]
    };
    myChart.setOption(option);

}
