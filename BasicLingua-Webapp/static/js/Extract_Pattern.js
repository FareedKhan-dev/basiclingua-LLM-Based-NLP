window.onload = function() {
    var cachedApiKey = localStorage.getItem("api_key");
    var cachedUserInput = localStorage.getItem("user_input");
    var cachedTargetLang = localStorage.getItem("patterns");
    

    document.getElementById("api_key").value = cachedApiKey || "";
    document.getElementById("user_input").value = cachedUserInput && "";
    document.getElementById("patterns").value = cachedTargetLang && "";
};



document.getElementById("extract-button").addEventListener("click", function() {
    var apiKey = document.getElementById("api_key").value.trim();
    var userInput = document.getElementById("user_input").value.trim();
    var patterns = document.getElementById("patterns").value.trim();

    if (apiKey === "" || userInput === "" || patterns === "") {
        alert("Please fill in all required fields.");
        return;
    }

    document.getElementById("loader").style.display = "block";

    localStorage.setItem("api_key", apiKey);
    localStorage.setItem("user_input", userInput);
    localStorage.setItem("patterns", patterns);

    $.ajax({
        url: translationUrl,
        type: "POST",
        data: $("#translation-form").serialize(),
        success: function(response) {
            document.getElementById("loader").style.display = "none";
            console.log("Response:", response);
            document.getElementById("extraction-result").innerHTML = "<h4>Extraction Result</h4><p style='text-align: left;'>" + response.extracted_patterns + "</p>";
        },
        error: function(xhr, errmsg, err) {
            document.getElementById("loader").style.display = "none";
            document.getElementById("extraction-result").innerHTML = "<h4>Error retrieving output</h4><p style='text-align: left;'>" + xhr.status + "</p>";
            console.log("Error:", xhr.status);
        }
    });
});

document.getElementById("refresh-button").addEventListener("click", function() {
    document.getElementById("user_input").value = "";
    document.getElementById("patterns").value = "";
    document.getElementById("extraction-result").innerHTML = "";

    
    localStorage.removeItem("user_input");
    localStorage.removeItem("target_lang");
});
