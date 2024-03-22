window.onload = function() {
    
    var cachedApiKey = localStorage.getItem("api_key");
    var cachedUserInput = localStorage.getItem("user_input");
    var cachedTargetLang = localStorage.getItem("target_lang");

    
    document.getElementById("api_key").value = cachedApiKey || "";
    document.getElementById("user_input").value = cachedUserInput || "";
    document.getElementById("target_lang").value = cachedTargetLang || "";

    document.getElementById("translate-button").addEventListener("click", function() {
        var apiKey = document.getElementById("api_key").value.trim();
        var userInput = document.getElementById("user_input").value.trim();
        var targetLang = document.getElementById("target_lang").value.trim();

        if (apiKey === "" || userInput === "" || targetLang === "") {
            alert("Please fill in all required fields.");
            return;
        }

        document.getElementById("loader").style.display = "block";

        
        localStorage.setItem("api_key", apiKey);
        localStorage.setItem("user_input", userInput);
        localStorage.setItem("target_lang", targetLang);

        $.ajax({
            url: translationUrl,
            type: "POST",
            data: $("#translation-form").serialize(),
            success: function(response) {
                document.getElementById("loader").style.display = "none";
                console.log("Response:", response);
                document.getElementById("translation-result").innerHTML = "<h4>Translation Result</h4><p style='text-align: left;'>" + response.translated_text + "</p>";
            },
            error: function(xhr, errmsg, err) {
                document.getElementById("loader").style.display = "none";
                document.getElementById("translation-result").innerHTML = "<h4>Error retrieving output</h4><p style='text-align: left;'>" + xhr.status + "</p>"
                console.log("Error:", xhr.status + ": " + xhr.responseText);
            }
        });
    });

    document.getElementById("refresh-button").addEventListener("click", function() {
        document.getElementById("user_input").value = "";
        document.getElementById("target_lang").value = "";
        document.getElementById("translation-result").innerHTML = "";
    
        
        localStorage.removeItem("user_input");
        localStorage.removeItem("target_lang");
    });
};
