window.onload = function() {
    var cachedApiKey = localStorage.getItem("api_key");
    var cachedUserInput = localStorage.getItem("user_input");
    var cachedTargetLang = localStorage.getItem("replacement_rules");

    
    document.getElementById("api_key").value = cachedApiKey || "";
    document.getElementById("user_input").value = cachedUserInput && "";
    document.getElementById("replacement_rules").value = cachedTargetLang && "";
};

document.getElementById("process-button").addEventListener("click", function() {
    var apiKey = document.getElementById("api_key").value.trim();
    var userInput = document.getElementById("user_input").value.trim();
    var replacementRules = document.getElementById("replacement_rules").value.trim();

    if (apiKey === "" || userInput === "" || replacementRules === "") {
        alert("Please fill in all required fields.");
        return;
    }

    document.getElementById("loader").style.display = "block";

    localStorage.setItem("api_key", apiKey);
    localStorage.setItem("user_input", userInput);
    localStorage.setItem("replacment_rules", replacementRules);

    $.ajax({
        url: translationUrl,
        type: "POST",
        data: $("#translation-form").serialize(),
        success: function(response) {
            document.getElementById("loader").style.display = "none";
            console.log("Response:", response);
            document.getElementById("processed-result").innerHTML = "<h4>Text Replacement Result</h4><p style='text-align: left;'>" + response.answer + "</p>";
        },
        error: function(xhr, errmsg, err) {
            document.getElementById("loader").style.display = "none";
            document.getElementById("processed-result").innerHTML = "<h4>Error retrieving output</h4><p style='text-align: left;'>" + xhr.status + "</p>";
            console.log("Error:", xhr.status + ": " + xhr.responseText);
        }
    });
});

    document.getElementById("refresh-button").addEventListener("click", function() {
        document.getElementById("user_input").value = "";
        document.getElementById("replacement_rules").value = "";
        document.getElementById("processed-result").innerHTML = "";

        
        localStorage.removeItem("user_input");
        localStorage.removeItem("target_lang");
    });