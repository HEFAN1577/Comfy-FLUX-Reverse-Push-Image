import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Comfy.GroqPromptReverse",
    async setup() {
        // 保存原始的getNodeMenuOptions函数
        const origGetNodeMenuOptions = LGraphCanvas.prototype.getNodeMenuOptions;
        
        // 重写getNodeMenuOptions函数
        LGraphCanvas.prototype.getNodeMenuOptions = function(node) {
            const options = origGetNodeMenuOptions.call(this, node);
            
            if (node.type === "GroqPromptReverse") {
                // 找到API密钥输入框并设置为密码类型
                const widgets = node.widgets || [];
                const apiKeyWidget = widgets.find(w => w.name === "api_key");
                if (apiKeyWidget) {
                    apiKeyWidget.password = true;
                    apiKeyWidget.type = "password";
                }

                // 添加设置API密钥的菜单选项
                options.push({
                    content: "Set Groq API Key",
                    callback: () => {
                        const key = prompt("Enter your Groq API key:");
                        if (key) {
                            localStorage.setItem("groq_api_key", key);
                            // 如果存在API密钥输入框，更新其值
                            if (apiKeyWidget) {
                                apiKeyWidget.value = key;
                                node.setDirtyCanvas(true);
                            }
                        }
                    }
                });
            }
            return options;
        };
    }
}); 