package coze

import (
	"fmt"
	"github.com/bincooo/chatgpt-adapter/internal/common"
	"github.com/bincooo/chatgpt-adapter/internal/gin.handler/response"
	"github.com/bincooo/chatgpt-adapter/internal/plugin"
	"github.com/bincooo/chatgpt-adapter/logger"
	"github.com/bincooo/coze-api"
	"github.com/gin-gonic/gin"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
	"encoding/json"
	"os"
)

type Config struct {
    Models map[string][]ModelConfig `json:"models"`
}

type ModelConfig struct {
    Cookie     string `json:"cookie"`
    Model      string `json:"model"`
    Used       int    `json:"used"`
    StartTime  int64  `json:"start_time"`
    Lock       int    `json:"lock"`
}

var configMutex = &sync.Mutex{}

func readConfig() (Config, error) {
    configMutex.Lock()
    defer configMutex.Unlock()

    var config Config
    configFile, err := os.ReadFile("config.json")
    if err != nil {
        return config, err
    }

    err = json.Unmarshal(configFile, &config)
    return config, err
}

func updateConfig(config Config) error {
    configMutex.Lock()
    defer configMutex.Unlock()

    configFile, err := json.MarshalIndent(config, "", "  ")
    if err != nil {
        return err
    }

    return os.WriteFile("config.json", configFile, 0644)
}

func selectAndLockConfig(modelType string) (*ModelConfig, Config, error) {
    config, err := readConfig()
    if err != nil {
        return nil, config, err
    }

    // This dynamically accesses the correct slice of configurations based on modelType
    availableConfigs, ok := config.Models[modelType]
    if !ok {
        return nil, config, fmt.Errorf("model type %s not found in config", modelType)
    }

    var selectedConfig *ModelConfig
    currentTime := time.Now().Unix()
    oneDayInSeconds := int64(86400)

    for i := range availableConfigs {
        if availableConfigs[i].Used > 50 {
            continue // Skip this configuration because it's been used more than 50 times.
        }

        if currentTime-availableConfigs[i].StartTime > oneDayInSeconds {
            // More than 1 day has passed since start_time, reset start_time and used count.
            availableConfigs[i].StartTime = currentTime
            availableConfigs[i].Used = 0
        }

        if availableConfigs[i].Lock == 0 {
            availableConfigs[i].Lock = 1 // Lock the configuration
            availableConfigs[i].Used++   // Increment the used count
            selectedConfig = &availableConfigs[i]
            break
        }
		if availableConfigs[i].Lock == 1 {
            availableConfigs[i].Lock = 1 // Lock the configuration
            availableConfigs[i].Used++   // Increment the used count
            selectedConfig = &availableConfigs[i]
            break
        }
    }

    if selectedConfig == nil {
        return nil, config, nil // No available configuration
    }

    err = updateConfig(config)
    if err != nil {
        return nil, config, err
    }

    return selectedConfig, config, nil
}


var (
	Adapter = API{}
	Model   = "coze"

	// 35-16k
	botId35_16k   = "7353052833752694791"
	version35_16k = "1716683639615"
	scene35_16k   = 2

	// 8k
	botId8k   = "7353047124357365778"
	version8k = "1716940640540"
	scene8k   = 2

	// 128k
	botId128k   = "7353048532129644562"
	version128k = "1716940665830"
	scene128k   = 2

	mu    sync.Mutex
	rwMus = make(map[string]*common.ExpireLock)
)

type API struct {
	plugin.BaseAdapter
}

func (API) Match(ctx *gin.Context, model string) bool {
	if Model == model {
		return true
	}

	if strings.HasPrefix(model, "coze/") {
		// coze/botId-version-scene
		values := strings.Split(model[5:], "-")
		if len(values) > 2 {
			_, err := strconv.Atoi(values[2])
			return err == nil
		}
	}

	token := ctx.GetString("token")
	if model == "dall-e-3" {
		if strings.Contains(token, "msToken=") || strings.Contains(token, "sessionid=") {
			return true
		}
	}
	return false
}

func (API) Models() []plugin.Model {
	return []plugin.Model{
		{
			Id:      Model,
			Object:  "model",
			Created: 1686935002,
			By:      Model + "-adapter",
		},
	}
}

func (API) Completion(ctx *gin.Context) {
	var (
		cookie     = ctx.GetString("token")
		proxies    = ctx.GetString("proxies")
		notebook   = ctx.GetBool("notebook")
		completion = common.GetGinCompletion(ctx)
		matchers   = common.GetGinMatchers(ctx)
	)
	modelType := completion.Model
	selectedConfig, config, err := selectAndLockConfig(modelType)
    if err != nil {
        logger.Error(err)
        response.Error(ctx, -1, err.Error())
        return
    }

    if selectedConfig == nil {
        response.Error(ctx, -1, "No available configuration for model "+modelType)
        return
    }
	cookie = selectedConfig.Cookie
    completion.Model = selectedConfig.Model
	if plugin.NeedToToolCall(ctx) {
		if completeToolCalls(ctx, cookie, proxies, completion) {
			return
		}
	}

	pMessages, tokens, err := mergeMessages(ctx)
	if err != nil {
		logger.Error(err)
		response.Error(ctx, -1, err)
		return
	}

	ctx.Set(ginTokens, tokens)
	options := newOptions(proxies, completion.Model, pMessages)
	co, msToken := extCookie(cookie)
	chat := coze.New(co, msToken, options)

	var lock *common.ExpireLock
	if isOwner(completion.Model) {
		var system string
		message := pMessages[0]
		if message.Role == "system" {
			system = message.Content
		}

		var value map[string]interface{}
		value, err = chat.BotInfo(ctx.Request.Context())
		if err != nil {
			logger.Error(err)
			response.Error(ctx, -1, err)
			return
		}

		// 加锁
		botId := customBotId(completion.Model)
		lock = newLock(botId)
		if !lock.Lock(ctx.Request.Context()) {
			// 上锁失败
			logger.Errorf("上锁失败：%s", botId)
			response.Error(ctx, http.StatusTooManyRequests, "Too Many Requests")
			return
		}

		logger.Infof("上锁成功：%s", botId)
		if err = chat.DraftBot(ctx.Request.Context(), coze.DraftInfo{
			Model:            value["model"].(string),
			TopP:             completion.TopP,
			Temperature:      completion.Temperature,
			MaxTokens:        completion.MaxTokens,
			FrequencyPenalty: 0,
			PresencePenalty:  0,
			ResponseFormat:   0,
		}, system); err != nil {
			// 全局配置修改失败，解锁
			lock.Unlock()
			rmLock(botId)
			logger.Error(fmt.Errorf("全局配置修改失败，解锁：%s， %v", botId, err))
			response.Error(ctx, -1, err)
			return
		}
	}

	query := ""
	if notebook && len(pMessages) > 0 {
		// notebook 模式只取第一条 content
		query = pMessages[0].Content
	} else {
		query = coze.MergeMessages(pMessages)
	}

	chatResponse, err := chat.Reply(ctx.Request.Context(), coze.Text, query)
	// 构建完请求即可解锁
	if lock != nil {
		lock.Unlock()
		botId := customBotId(completion.Model)
		rmLock(botId)
		logger.Infof("构建完成解锁：%s", botId)
	}
	
	selectedConfig.Lock = 0
	if err := updateConfig(config); err != nil {
        logger.Error(err)
        // Handle error (optional)
    }

	if err != nil {
		logger.Error(err)
		response.Error(ctx, -1, err)
		return
	}

	// 自定义标记块中断
	cancel, matcher := common.NewCancelMather(ctx)
	matchers = append(matchers, matcher)

	content := waitResponse(ctx, matchers, cancel, chatResponse, completion.Stream)
	if content == "" && response.NotResponse(ctx) {
		response.Error(ctx, -1, "EMPTY RESPONSE")
	}
}

func (API) Generation(ctx *gin.Context) {
	var (
		cookie     = ctx.GetString("token")
		proxies    = ctx.GetString("proxies")
		generation = common.GetGinGeneration(ctx)
	)

	// 只绘画用3.5 16k即可
	options := coze.NewDefaultOptions(botId35_16k, version35_16k, scene35_16k, false, proxies)
	co, msToken := extCookie(cookie)
	chat := coze.New(co, msToken, options)
	image, err := chat.Images(ctx.Request.Context(), generation.Message)
	if err != nil {
		logger.Error(err)
		response.Error(ctx, -1, err)
		return
	}

	if (generation.Size == "HD" || strings.HasPrefix(generation.Size, "1792x")) && common.HasMfy() {
		v, e := common.Magnify(ctx, image)
		if e != nil {
			logger.Error(e)
		} else {
			image = v
		}
	}

	ctx.JSON(http.StatusOK, gin.H{
		"created": time.Now().Unix(),
		"styles:": make([]string, 0),
		"data": []map[string]string{
			{"url": image},
		},
	})
}

func newLock(token string) *common.ExpireLock {
	mu.Lock()
	defer mu.Unlock()
	if m, ok := rwMus[token]; ok {
		return m
	}

	m := common.NewExpireLock()
	rwMus[token] = m
	return m
}

func rmLock(token string) {
	mu.Lock()
	defer mu.Unlock()
	if m, ok := rwMus[token]; ok {
		if m.IsIdle() {
			delete(rwMus, token)
		}
	}
}

func customBotId(model string) string {
	if strings.HasPrefix(model, "coze/") {
		values := strings.Split(model[5:], "-")
		return values[0]
	}
	return ""
}

func newOptions(proxies string, model string, pMessages []coze.Message) (options coze.Options) {
	if strings.HasPrefix(model, "coze/") {
		values := strings.Split(model[5:], "-")
		scene, err := strconv.Atoi(values[2])
		if err == nil {
			options = coze.NewDefaultOptions(values[0], values[1], scene, isOwner(model), proxies)
			logger.Infof("using custom coze options: botId = %s, version = %s, scene = %d", values[0], values[1], scene)
			return
		}
		logger.Error(err)
	}

	options = coze.NewDefaultOptions(botId8k, version8k, scene8k, false, proxies)
	// 大于7k token 使用 gpt-128k
	if token := calcTokens(pMessages); token > 7000 {
		options = coze.NewDefaultOptions(botId128k, version128k, scene128k, false, proxies)
	}

	return
}

func extCookie(co string) (cookie, msToken string) {
	cookie = co
	index := strings.Index(cookie, "[msToken=")
	if index > -1 {
		end := strings.Index(cookie[index:], "]")
		if end > -1 {
			msToken = cookie[index+6 : index+end]
			cookie = cookie[:index] + cookie[index+end+1:]
		}
	}
	return
}

func isOwner(model string) bool {
	return strings.HasSuffix(model, "-o")
}
