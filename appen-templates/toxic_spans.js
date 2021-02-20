var require = require || function(what,callback) {
  var myScript = Asset.javascript('//ajax.googleapis.com/ajax/libs/jquery/1.11.3/jquery.min.js', {
    onLoad: function(){
      jQuery.noConflict();
      callback(null,null,jQuery);
    }
  });
};

require(['jquery-noconflict'], function(jQuery) {
  var $ = jQuery || window.jQuery;
  var lastSelection;
  document.addEventListener("selectionchange", function() {
    lastSelection = window.getSelection();
  });
  var testQuestions = document.location.pathname.match(/test_questions/);
  var klass = testQuestions ? ".spans_gold" : ".spans";
  
  $(".spans").prop("readonly", true);
  
  //Display the remove button for the first row, and change it's behaviour
  $(klass).each(function(i,e){
    $(this).after('<a href="#"></a>').next().css({
      display:"inline-block", backgroundImage: "url(/assets/remove_with_hover.png)", 
      height: "19px", width: "19px"}).on('click', function(){
        $(this).prev().val("");
        //this code removes the first span tag created to highlight what users have clicked on:
        var previd = $(this).prev().attr('id');
        var myElements = document.getElementsByClassName(previd);
        for (var i = 0; i < myElements.length; i++) {
            myElements[i].style.backgroundColor = "";
            myElements[i].classList.remove("chosen");
        }
        var row = $(this).closest('.entity_extract');
        row.find('.history').empty();
        var parent = row.parent();
        var textList = parent.find(klass);
        if(textList.length > 1) {
          textList.each(function(index){
          if (index < textList.length){
            $(this).val(textList.eq(index+1).val());
            var newId = textList.eq(index+1).attr('id');
            $(this).attr("id", newId);
          }
        });
        textList.last().parent().remove();
        }
        
        return false;
    });
    
  });
  // just shift everything up and remove last row when doing this. 
  function getSelectionCharacterOffsetsWithin(element, btnColor) {
    var startOffset = 0, endOffset = 0, selectedText = "null";
    if (typeof window.getSelection != "undefined") {
        var selection = window.getSelection();
        selectedText = selection.toString();
        var range = selection.getRangeAt(0);
        console.log(range)
        //Strip trailing punctation
        selectedText = selectedText.replace(/[\s.,!?:;'"-]+$/, "");
        //Leading space / quotes
        var offset = 0
        var match = selectedText.match(/^[\s"']+/);
        if(match)
          offset = match.length;
          console.log(offset)
          selectedText = selectedText.replace(/^[\s"']+/,"");
        if (selectedText === ""){
          alert("Error: you must select at least one character");
          startOffset = 0, endOffset = 0, selectedText = "null";
        }
        else{  
        var preCaretRange = range.cloneRange();
        preCaretRange.selectNodeContents(element);
        preCaretRange.setEnd(range.startContainer, range.startOffset);
        startOffset = preCaretRange.toString().length;
        startOffset = startOffset + offset;
        endOffset = startOffset + range.toString().length - 1;
        var newInputid = parseInt(Math.random() * 10000);
        //This is code to keep word highlighted after selecting
        var newNode = document.createElement("span");
        newNode.addClass('chosen');
        var previd = ("e" + newInputid);
        newNode.addClass(previd);
        newNode.appendChild(range.extractContents());
        range.insertNode(newNode);
        var textSegment = $("." + previd);
        textSegment[0].style.backgroundColor = btnColor;
        }
        
    }
    return { start: startOffset, end: endOffset, text: selectedText, cid: previd };
}
  
  $('.entity_types button').on('click', function(e) {
    e.preventDefault();
    if(lastSelection.rangeCount === 0) {
      return alert("You must select text above before choosing.");
    }
    var sent_text_area = $(this).closest('.entity_extract').find('.passage').show()[0];
    var btnColor = $(this)[0].style.color;
    var selOffsets = getSelectionCharacterOffsetsWithin(sent_text_area, btnColor);
    var selectedText = selOffsets.text;
    var startOffset = selOffsets.start;
    var endOffset = startOffset + selectedText.length - 1;
    if (selectedText != "null"){
    var formElement = $(this).closest('.cml'); 
    var textInputElement = $(formElement).find(klass).last();
    var category = $(this).text();

    console.log(selectedText)
    

    //Insert Input Box
    if(textInputElement.val().length > 0) {
      new Element(textInputElement.nextAll(".multiple_add").get(0)).fireEvent('click');
      textInputElement = $(formElement).find(klass).last();
      textInputElement.next().css({display: "none"});
    }
    
    //code if using broken up text
    //add sent start value to selected value to get selection start from full paragraph. 
    //var sent_start = $(this).closest('.entity_extract').find('.hidden').text();
    //var total = startOffset+parseInt(sent_start);
    
    //Put data in text box
      textInputElement.val(startOffset + ":" + category + ":"  + selectedText);
      textInputElement.attr("id", selOffsets.cid);
      textInputElement.nextAll(".multiple_remove").on('click', function() {
        //this code removes the span tag created to highlight what ppl click on for everything but the first one
       //var textSegment = $("." + selOffsets.cid);
       var elements = textInputElement.closest('.cml_row').find('input').attr('id');
       console.log(selOffsets.cid);
       console.log(elements);
        var myElements = document.getElementsByClassName(elements);
        for (var i = 0; i < myElements.length; i++) {
            myElements[i].style.backgroundColor = "";
        }
        textInputElement.val("");
    });
    }
  });
});
