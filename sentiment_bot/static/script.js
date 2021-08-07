var sentence = "";
var splitted_sentence = "";
var target = "";
var targetWord = "";

var url = window.location.href+"analysis/?"
url = url.replace("http:", "")
const sendHandler = ()=>{
    let text = document.getElementById('user_input').value;
    document.getElementById('user_input').value = ""
    let Http = new XMLHttpRequest();
    display("user", text);
    if(!sentence){
        sentence = text.toLowerCase().replace(/[^\w\s]|_/g, "").replace(/\s{2,}/g," ");
        splitted_sentence = sentence.split(" ")
        display("bot", "What is your target word?");
        
    } else{
        parsedText = text.toLowerCase().trim()
        index = splitted_sentence.indexOf(parsedText)
        if(index==-1){
            message = "We cannot find your target word \"<i>" + text + "</i>\" in the sentence. Please enter again."
            display("bot", message)
            return;
        }
        target = index;
        targetWord = splitted_sentence[target];
        sentenceWithSpace =  encodeURIComponent(sentence.trim())
        var composedUrl = url + "sentence=" + sentenceWithSpace + "&target="+target
        Http.onreadystatechange = function() { 
            if(Http.readyState == 4 && Http.status == 200){
                result = JSON.parse(Http.responseText)
            
                var message = "I think the sentiment of \"<i>" + targetWord+ "</i>\" in your text: \"<i>" + sentence + "</i>\" is <b>" + result.sentiment + "</b>."
                if(result){
                    display("bot", message)
                    message = "And I am <b>" + result.confidence + "%</b> sure about that."
                    display("bot", message)
                    reset();
                    return
                }
            }
           
        }
        
        Http.open("GET", composedUrl)
        Http.send(null); 
        
    }
}

const display = (speaker, message) =>{
    if(message){
        var new_message;
        var prev = document.getElementById('chat').innerHTML;
        if (speaker == "bot"){
           
            new_message = `
            <div class="balon2">
                <a >   ` + message + `    </a>
            </div>
    
            `
        }
        else{
            new_message = `
            <div class="balon1 float-right">
                <a class="float-right">  ` + message + `   </a>
            </div>      
            `
        }
        document.getElementById('chat').innerHTML = prev + new_message;
    }
    }
    

const reset = () =>{
    sentence = ""
    display("bot", "What is your text?")
}