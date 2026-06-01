package com.cartridgestamper

import android.app.ActivityOptions
import android.graphics.Color as AndroidColor
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Typeface
import android.hardware.display.DisplayManager
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.MenuDefaults
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.documentfile.provider.DocumentFile
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.delay
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.nio.FloatBuffer
import java.util.Locale
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

private const val TEMPLATE_ASSET_DIR = "templates"
private const val MASK_INSET_PIXELS = 5
private const val GAME_CODE_LOOKUP_PLACEHOLDER = "Looking up code..."
private const val APP_VERSION_FALLBACK = "v.1.5.0"
private const val LAMA_MODEL_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx?download=true"
private const val LAMA_MODEL_NAME = "lama_fp32.onnx"
const val ACTION_LOWER_PREVIEW = "com.cartridgestamper.LOWER_PREVIEW"
const val ACTION_CLOSE_LOWER_DISPLAY = "com.cartridgestamper.CLOSE_LOWER_DISPLAY"
const val ACTION_LOWER_PREVIEW_TAPPED = "com.cartridgestamper.LOWER_PREVIEW_TAPPED"
const val ACTION_LOWER_PREVIEW_GESTURE = "com.cartridgestamper.LOWER_PREVIEW_GESTURE"
const val EXTRA_PREVIEW_COMMAND = "preview_command"
const val EXTRA_BITMAP_BYTES = "bitmap_bytes"
const val EXTRA_BITMAP_FILE = "bitmap_file"
const val EXTRA_GESTURE_SCALE = "gesture_scale"
const val EXTRA_GESTURE_DX = "gesture_dx"
const val EXTRA_GESTURE_DY = "gesture_dy"
const val COMMAND_UPDATE_LOWER_PREVIEW = "update_lower_preview"

private val AppBackground = Color(0xFF07090D)
private val PanelBackground = Color(0xFF0B141A)
private val ControlBackground = Color(0xFF101927)
private val TrackBackground = Color(0xFF0F3E66)
private val Teal = Color(0xFF28C9FF)
private val Blue = Color(0xFF2492E8)
private val HotPink = Color(0xFFFF3EA5)
private val TextPrimary = Color(0xFFF0F7FA)
private val TextSecondary = Color(0xFF9CB3BE)

private val lamaRuntimeLock = Any()
private var lamaOrtEnv: OrtEnvironment? = null
private var lamaOrtSession: OrtSession? = null
private var lamaOrtModelPath: String? = null

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            CartridgeStamperTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    StamperScreen()
                }
            }
        }
    }

    override fun onDestroy() {
        closeLowerDisplayPreview(this)
        super.onDestroy()
    }
}


private fun Bitmap.toCocoonSquareIconCanvas(
    canvasSize: Int = 1024,
    visibleHeightRatio: Float = 0.98f
): Bitmap {
    if (width == canvasSize && height == canvasSize) return this

    val output = Bitmap.createBitmap(canvasSize, canvasSize, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(output)
    canvas.drawColor(AndroidColor.TRANSPARENT)

    val maxHeight = canvasSize * visibleHeightRatio
    val maxWidth = canvasSize * 0.98f
    val scale = minOf(maxWidth / width.toFloat(), maxHeight / height.toFloat())

    val drawWidth = width * scale
    val drawHeight = height * scale
    val left = (canvasSize - drawWidth) / 2f
    val top = (canvasSize - drawHeight) / 2f

    val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG or Paint.DITHER_FLAG)
    canvas.drawBitmap(this, null, RectF(left, top, left + drawWidth, top + drawHeight), paint)

    return output
}



private fun Canvas.clearGameCubeInnerDiscCutout(width: Int, height: Int) {
    val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        xfermode = PorterDuffXfermode(PorterDuff.Mode.CLEAR)
    }

    // GameCube disc templates have a center hub. This clears the innermost circle
    // so lower media cannot show through the hole.
    val cx = width / 2f
    val cy = height / 2f
    val radius = minOf(width, height) * 0.118f
    drawCircle(cx, cy, radius, paint)
    paint.xfermode = null
}


@Composable
private fun StamperScreen() {
    val context = LocalContext.current

    val templates = remember { loadTemplateAssets(context) }
    var selectedTemplate by remember { mutableStateOf(templates.firstOrNull()) }
    var selectedMediaKind by remember { mutableStateOf(MediaFolderKind.PhysicalMedia) }
    var coverImages by remember { mutableStateOf(emptyList<ImageAsset>()) }
    var selectedCover by remember { mutableStateOf<ImageAsset?>(null) }
    var extraLayerEnabled by remember { mutableStateOf(false) }
    var extraLayerImages by remember { mutableStateOf(emptyList<ImageAsset>()) }
    var extraLayerCover by remember { mutableStateOf<ImageAsset?>(null) }
    var extraLayerActiveFolder by remember { mutableStateOf(selectedTemplate?.defaultMediaDirectory(selectedMediaKind)?.displayPath().orEmpty()) }
    var extraLayerOffsetX by remember { mutableFloatStateOf(0f) }
    var extraLayerOffsetY by remember { mutableFloatStateOf(0f) }
    var extraLayerScale by remember { mutableFloatStateOf(100f) }
    var activeCoverFolder by remember { mutableStateOf(selectedTemplate?.defaultMediaDirectory(selectedMediaKind)?.displayPath().orEmpty()) }
    var offsetX by remember { mutableFloatStateOf(0f) }
    var offsetY by remember { mutableFloatStateOf(0f) }
    var coverScale by remember { mutableFloatStateOf(100f) }
    var showTemplateOverlay by remember { mutableStateOf(true) }
    var outpaintArtwork by remember { mutableStateOf(false) }
    var showGameBoyLogo by remember { mutableStateOf(true) }
    var logoScale by remember { mutableFloatStateOf(100f) }
    var logoOffsetX by remember { mutableFloatStateOf(0f) }
    var logoOffsetY by remember { mutableFloatStateOf(0f) }
    var show3dsSeal by remember { mutableStateOf(true) }
    var sealScale by remember { mutableFloatStateOf(100f) }
    var sealOffsetX by remember { mutableFloatStateOf(0f) }
    var sealOffsetY by remember { mutableFloatStateOf(0f) }
    var generateGameCode by remember { mutableStateOf(false) }
    var gameTitleQuery by remember { mutableStateOf("") }
    var gameCodeText by remember { mutableStateOf("") }
    var gameCodeBold by remember { mutableStateOf(false) }
    var gameCodeSizeStep by remember { mutableStateOf(0) }
    var gameCodeOffsetX by remember { mutableFloatStateOf(0f) }
    var gameCodeOffsetY by remember { mutableFloatStateOf(0f) }
    var gameCodeLookupStatus by remember { mutableStateOf<String?>(null) }
    var gameCodeLookupRequest by remember { mutableStateOf(0) }
    var outpaintFill by remember { mutableStateOf<OutpaintFill?>(null) }
    var outpaintStatus by remember { mutableStateOf<String?>(null) }
    var outpaintProgress by remember { mutableStateOf<OutpaintProgress?>(null) }
    var dualDisplayMode by remember { mutableStateOf(false) }
    var previewBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var artworkExportBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var status by remember { mutableStateOf("Select a template") }
    val appVersionLabel = remember(context) { appVersionLabel(context) }

    fun loadDefaultCoverFolder(template: TemplateAsset?) {
        val folder = template?.defaultMediaDirectory(selectedMediaKind)
        activeCoverFolder = folder?.displayPath().orEmpty()
        coverImages = folder?.let { loadImagesFromFolder(it) }.orEmpty()
        selectedCover = coverImages.firstOrNull()
        extraLayerActiveFolder = activeCoverFolder
        extraLayerImages = coverImages
        extraLayerCover = extraLayerImages.drop(1).firstOrNull() ?: extraLayerImages.firstOrNull()
        status = if (template == null) {
            "No templates bundled"
        } else {
            "Loaded ${coverImages.size} ${selectedMediaKind.label.lowercase(Locale.US)} files"
        }
    }

    val coverFolderLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocumentTree()
    ) { uri ->
        uri?.let {
            persistTreePermission(context, it)
            selectedMediaKind = MediaFolderKind.CustomFolder
            activeCoverFolder = "selected folder"
            coverImages = loadImagesFromTree(context, it)
            selectedCover = coverImages.firstOrNull()
            extraLayerActiveFolder = activeCoverFolder
            extraLayerImages = coverImages
            extraLayerCover = extraLayerImages.drop(1).firstOrNull() ?: extraLayerImages.firstOrNull()
            status = "Loaded ${coverImages.size} cover images"
        }
    }

    val coverImageLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        uri?.let {
            persistReadPermission(context, it)
            selectedMediaKind = MediaFolderKind.SpecificImage
            activeCoverFolder = "selected image"
            val image = imageAssetFromUri(context, it)
            coverImages = listOf(image)
            selectedCover = image
            extraLayerActiveFolder = activeCoverFolder
            extraLayerImages = coverImages
            extraLayerCover = null
            status = image.label
        }
    }

    val topLayerFolderLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocumentTree()
    ) { uri ->
        uri?.let {
            persistTreePermission(context, it)
            extraLayerActiveFolder = "selected top layer folder"
            extraLayerImages = loadImagesFromTree(context, it)
            extraLayerCover = extraLayerImages.firstOrNull()
            extraLayerEnabled = true
            status = "Loaded ${extraLayerImages.size} top layer images"
        }
    }

    val saveAsLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.CreateDocument("image/png")
    ) { uri ->
        val bitmap = artworkExportBitmap
        if (uri != null && bitmap != null) {
            writeBitmapToUri(context, bitmap, uri)
                .onSuccess { status = "Artwork export saved" }
                .onFailure { status = it.localizedMessage ?: "Export failed" }
        }
    }

    LaunchedEffect(selectedTemplate?.assetPath, selectedMediaKind) {
        if (selectedMediaKind.isDefaultSource) loadDefaultCoverFolder(selectedTemplate)
    }

    LaunchedEffect(selectedTemplate?.assetPath, selectedCover?.source?.key) {
        gameTitleQuery = selectedCover?.label?.cleanGameTitleForLookup().orEmpty()
        gameCodeText = ""
        gameCodeLookupStatus = null
    }

    LaunchedEffect(generateGameCode, selectedTemplate?.assetPath, selectedCover?.source?.key) {
        if (generateGameCode && selectedTemplate?.supportsGameCodeGeneration == true) {
            gameCodeLookupRequest++
        }
    }

    LaunchedEffect(extraLayerImages, extraLayerEnabled) {
        if (extraLayerEnabled && (extraLayerCover == null || extraLayerImages.none { it.source.key == extraLayerCover?.source?.key })) {
            extraLayerCover = extraLayerImages.firstOrNull()
        }
    }

    DisposableEffect(Unit) {
        val receiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context?, intent: Intent?) {
                when (intent?.action) {
                    ACTION_LOWER_PREVIEW_TAPPED -> {
                        dualDisplayMode = false
                        closeLowerDisplayPreview(context ?: return)
                    }

                    ACTION_LOWER_PREVIEW_GESTURE -> {
                        val scaleFactor = intent.getFloatExtra(EXTRA_GESTURE_SCALE, 1f)
                        val dx = intent.getFloatExtra(EXTRA_GESTURE_DX, 0f)
                        val dy = intent.getFloatExtra(EXTRA_GESTURE_DY, 0f)
                        if (scaleFactor.isFinite() && scaleFactor > 0f) {
                            coverScale = (coverScale * scaleFactor).coerceIn(5f, 220f)
                        }
                        if (dx.isFinite()) offsetX = (offsetX + dx).coerceIn(-500f, 500f)
                        if (dy.isFinite()) offsetY = (offsetY + dy).coerceIn(-500f, 500f)
                    }
                }
            }
        }
        val filter = IntentFilter().apply {
            addAction(ACTION_LOWER_PREVIEW_TAPPED)
            addAction(ACTION_LOWER_PREVIEW_GESTURE)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            context.registerReceiver(receiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            @Suppress("DEPRECATION")
            context.registerReceiver(receiver, filter)
        }
        onDispose {
            runCatching { context.unregisterReceiver(receiver) }
        }
    }

    LaunchedEffect(gameCodeLookupRequest) {
        if (gameCodeLookupRequest == 0) return@LaunchedEffect
        val template = selectedTemplate
        val cover = selectedCover
        val lookupTitle = gameTitleQuery.ifBlank { cover?.label?.cleanGameTitleForLookup().orEmpty() }
        if (generateGameCode && template?.supportsGameCodeGeneration == true && lookupTitle.isNotBlank()) {
            gameCodeLookupStatus = GAME_CODE_LOOKUP_PLACEHOLDER
            val foundCode = queryGameProductCode(
                titleQuery = lookupTitle,
                template = template,
                originalLabel = cover?.label
            )
            if (foundCode == null) {
                gameCodeLookupStatus = "Code not found. Enter it manually."
            } else {
                gameCodeText = foundCode
                gameCodeLookupStatus = "Found $foundCode"
            }
        } else {
            gameCodeLookupStatus = null
        }
    }

    LaunchedEffect(selectedTemplate?.assetPath, selectedCover?.source?.key, offsetX, offsetY, coverScale, outpaintArtwork) {
        outpaintFill = null
        outpaintProgress = null
        if (!outpaintArtwork) {
            outpaintStatus = null
            return@LaunchedEffect
        }

        val template = selectedTemplate
        val coverAsset = selectedCover
        if (template == null || coverAsset == null) {
            outpaintStatus = "Select cover art before Outpaint Fill"
            return@LaunchedEffect
        }

        val cover = decodeBitmap(context, coverAsset.source)
        val templateBitmap = decodeTemplateBitmap(context, template)
        if (cover == null) {
            outpaintStatus = "Unable to read cover art"
            return@LaunchedEffect
        }

        val width = templateBitmap?.width ?: cover.width
        val height = templateBitmap?.height ?: cover.height
        val artworkWindow = template.artworkWindow(context, width, height)
        if (artworkWindow == null) {
            outpaintStatus = "No template artwork window"
            return@LaunchedEffect
        }

        val coverRect = cover.coverDrawRect(width, height, offsetX, offsetY, coverScale)
        if (coverRect.contains(artworkWindow)) {
            outpaintStatus = "Outpaint not needed"
            return@LaunchedEffect
        }

        val key = outpaintRequestKey(template, coverAsset, offsetX, offsetY, coverScale)
        outpaintProgress = OutpaintProgress("Preparing outpaint", 0.01f)
        delay(350)
        val model = lamaModelFile(context)
        outpaintStatus = if (model.exists() && model.length() > 100_000_000L) {
            "Running LaMa fill"
        } else {
            "Downloading LaMa model"
        }

        runCatching {
            withContext(Dispatchers.IO) {
                generateLamaOutpaintForArtworkWindow(
                    context = context,
                    cover = cover,
                    artworkWindow = artworkWindow,
                    coverRect = coverRect
                ) { progress ->
                    withContext(Dispatchers.Main) {
                        outpaintProgress = progress
                        outpaintStatus = progress.label
                    }
                }
            }
        }.onSuccess {
            outpaintFill = OutpaintFill(key, it)
            outpaintProgress = OutpaintProgress("LaMa fill ready", 1f)
            outpaintStatus = "LaMa fill ready"
        }.onFailure {
            outpaintFill = null
            outpaintProgress = null
            outpaintStatus = "LaMa fill failed: ${it.message ?: "unknown error"}"
        }
    }

    val currentOutpaintKey = selectedTemplate?.let { template ->
        selectedCover?.let { cover -> outpaintRequestKey(template, cover, offsetX, offsetY, coverScale) }
    }

    LaunchedEffect(dualDisplayMode, previewBitmap) {
        if (dualDisplayMode) {
            val launchStatus = launchLowerDisplayPreview(context, previewBitmap)
            if (launchStatus != null) status = launchStatus
            previewBitmap?.let { sendLowerPreviewUpdate(context, it) }
        } else {
            closeLowerDisplayPreview(context)
        }
    }

    LaunchedEffect(selectedTemplate?.assetPath, selectedCover?.source?.key, extraLayerEnabled, extraLayerCover?.source?.key, offsetX, offsetY, coverScale, extraLayerOffsetX, extraLayerOffsetY, extraLayerScale, showTemplateOverlay, outpaintArtwork, outpaintFill?.key, showGameBoyLogo, show3dsSeal, sealOffsetX, sealOffsetY, sealScale, logoOffsetX, logoOffsetY, logoScale, generateGameCode, gameCodeText, gameCodeBold, gameCodeSizeStep, gameCodeOffsetX, gameCodeOffsetY) {
        val templateAsset = selectedTemplate
        val coverAsset = selectedCover
        val topLayerAsset = extraLayerCover?.takeIf { extraLayerEnabled }
        val activeGameCode = gameCodeText.normalizeManualGameCode().takeIf { generateGameCode && it.isNotBlank() }
        val validOutpaintFill = outpaintFill?.takeIf { it.key == currentOutpaintKey }?.bitmap
        val drawGameBoyLogo = showGameBoyLogo
        val draw3dsSeal = show3dsSeal
        val drawSealOffsetX = sealOffsetX
        val drawSealOffsetY = sealOffsetY
        val drawSealScale = sealScale
        val drawLogoOffsetX = logoOffsetX
        val drawLogoOffsetY = logoOffsetY
        val drawLogoScale = logoScale
        val drawGameCodeBold = gameCodeBold
        val drawGameCodeSizeStep = gameCodeSizeStep
        val drawGameCodeOffsetX = gameCodeOffsetX
        val drawGameCodeOffsetY = gameCodeOffsetY
        val drawOffsetX = offsetX
        val drawOffsetY = offsetY
        val drawCoverScale = coverScale
        val drawExtraOffsetX = extraLayerOffsetX
        val drawExtraOffsetY = extraLayerOffsetY
        val drawExtraScale = extraLayerScale
        val drawTemplateOverlay = showTemplateOverlay
        val drawOutpaintArtwork = outpaintArtwork
        delay(90)
        val rendered = withContext(Dispatchers.Default) {
            val templateMask = templateAsset?.let { decodeTemplateMask(context, it) }
            val preparedTemplate = templateAsset?.let {
                prepareTemplateBitmap(
                    context = context,
                    template = it,
                    showGameBoyLogo = drawGameBoyLogo,
                    show3dsSeal = draw3dsSeal,
                    sealOffsetX = drawSealOffsetX,
                    sealOffsetY = drawSealOffsetY,
                    sealScale = drawSealScale,
                    logoOffsetX = drawLogoOffsetX,
                    logoOffsetY = drawLogoOffsetY,
                    logoScale = drawLogoScale,
                    gameProductCode = activeGameCode,
                    gameCodeBold = drawGameCodeBold,
                    gameCodeSizeStep = drawGameCodeSizeStep,
                    gameCodeOffsetX = drawGameCodeOffsetX,
                    gameCodeOffsetY = drawGameCodeOffsetY
                )
            }
            val coverBitmap = coverAsset?.let { decodeBitmap(context, it.source) }
            val extraBitmap = topLayerAsset?.let { decodeBitmap(context, it.source) }
            val preview = renderComposite(
                context = context,
                templateAsset = templateAsset,
                template = preparedTemplate,
                artworkMask = templateMask,
                cover = coverBitmap,
                extraCover = extraBitmap,
                offsetXPx = drawOffsetX,
                offsetYPx = drawOffsetY,
                scalePercent = drawCoverScale,
                extraOffsetXPx = drawExtraOffsetX,
                extraOffsetYPx = drawExtraOffsetY,
                extraScalePercent = drawExtraScale,
                outpaintArtwork = drawOutpaintArtwork,
                outpaintFill = validOutpaintFill,
                drawTemplateOverlay = drawTemplateOverlay
            )
            val artwork = renderComposite(
                context = context,
                templateAsset = templateAsset,
                template = preparedTemplate,
                artworkMask = templateMask,
                cover = coverBitmap,
                extraCover = extraBitmap,
                offsetXPx = drawOffsetX,
                offsetYPx = drawOffsetY,
                scalePercent = drawCoverScale,
                extraOffsetXPx = drawExtraOffsetX,
                extraOffsetYPx = drawExtraOffsetY,
                extraScalePercent = drawExtraScale,
                outpaintArtwork = drawOutpaintArtwork,
                outpaintFill = validOutpaintFill,
                drawTemplateOverlay = false
            )
            preview to artwork
        }
        previewBitmap = rendered.first
        artworkExportBitmap = rendered.second
    }

    val step1Section: @Composable () -> Unit = {
        if (needsAllFilesAccess()) {
            Button(
                onClick = { openAllFilesSettings(context) },
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(containerColor = HotPink)
            ) {
                Text("Grant Files Access")
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            StepLabel("1", "Select Cartridge")
            Text(
                text = appVersionLabel,
                style = MaterialTheme.typography.labelMedium,
                color = Color(0xFF4A545E),
                fontWeight = FontWeight.SemiBold
            )
        }
        TemplateDropdown(
            templates = templates,
            selected = selectedTemplate,
            onSelect = {
                selectedTemplate = it
                offsetX = 0f
                offsetY = 0f
                coverScale = 100f
                showTemplateOverlay = true
                extraLayerEnabled = false
                extraLayerImages = coverImages
                extraLayerActiveFolder = activeCoverFolder
                extraLayerCover = extraLayerImages.drop(1).firstOrNull() ?: extraLayerImages.firstOrNull()
                extraLayerOffsetX = 0f
                extraLayerOffsetY = 0f
                extraLayerScale = 100f
                outpaintArtwork = false
                outpaintFill = null
                outpaintStatus = null
                outpaintProgress = null
                logoScale = 100f
                logoOffsetX = 0f
                logoOffsetY = 0f
                sealScale = 100f
                sealOffsetX = 0f
                sealOffsetY = 0f
                generateGameCode = false
                gameTitleQuery = selectedCover?.label?.cleanGameTitleForLookup().orEmpty()
                gameCodeText = ""
                gameCodeBold = false
                gameCodeSizeStep = 0
                gameCodeOffsetX = 0f
                gameCodeOffsetY = 0f
                gameCodeLookupStatus = null
            }
        )
    }

    val step2Section: @Composable () -> Unit = {
        StepLabel("2", "Cover Art Source")
        MediaFolderDropdown(
            selected = selectedMediaKind,
            onSelect = {
                when (it) {
                    MediaFolderKind.CustomFolder -> coverFolderLauncher.launch(null)
                    MediaFolderKind.SpecificImage -> coverImageLauncher.launch(arrayOf("image/*"))
                    else -> selectedMediaKind = it
                }
            }
        )
        PathLine(
            "Folder",
            selectedTemplate?.defaultMediaDirectory(selectedMediaKind)?.displayPath() ?: activeCoverFolder
        )
        Text(
            text = activeCoverFolder,
            style = MaterialTheme.typography.labelSmall,
            color = TextSecondary,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
        AssetList(
            assets = coverImages,
            selected = selectedCover,
            emptyText = "No media in this folder",
            onSelect = {
                selectedCover = it
                status = it.label
            }
        )
    }

    val step3Section: @Composable () -> Unit = {
        StepLabel("3", "Alignment")
        ToggleRow("Add Layer", extraLayerEnabled) {
            extraLayerEnabled = it
            if (it && extraLayerCover == null) {
                extraLayerCover = extraLayerImages.firstOrNull()
            }
        }
        if (extraLayerEnabled) {
            Text("Top Layer", style = MaterialTheme.typography.labelMedium, color = HotPink, fontWeight = FontWeight.SemiBold)
            PathLine("Folder", extraLayerActiveFolder)
            OutlinedButton(
                onClick = { topLayerFolderLauncher.launch(null) },
                modifier = Modifier.fillMaxWidth(),
                border = BorderStroke(1.dp, Blue)
            ) {
                Text("Choose Top Layer Folder")
            }
            AssetList(
                assets = extraLayerImages,
                selected = extraLayerCover,
                emptyText = "No top layer media",
                onSelect = {
                    extraLayerCover = it
                    status = "Top Layer: ${it.label}"
                }
            )
            ControlSlider("Layer Scale", extraLayerScale, 5f..220f) { extraLayerScale = it.roundToInt().toFloat() }
            ControlSlider("Layer X", extraLayerOffsetX, -500f..500f) { extraLayerOffsetX = it.roundToInt().toFloat() }
            ControlSlider("Layer Y", extraLayerOffsetY, -500f..500f) { extraLayerOffsetY = it.roundToInt().toFloat() }
            HorizontalDivider(color = Blue.copy(alpha = 0.35f))
        }
        Text("Bottom Layer", style = MaterialTheme.typography.labelMedium, color = TextPrimary, fontWeight = FontWeight.SemiBold)
        ToggleRow("Show Template", showTemplateOverlay) { showTemplateOverlay = it }
        ControlSlider("Scale", coverScale, 5f..220f) { coverScale = it.roundToInt().toFloat() }
        ControlSlider("X offset", offsetX, -500f..500f) { offsetX = it.roundToInt().toFloat() }
        ControlSlider("Y offset", offsetY, -500f..500f) { offsetY = it.roundToInt().toFloat() }
        ToggleRow("Outpaint Fill", outpaintArtwork) { outpaintArtwork = it }
        outpaintProgress?.takeIf { outpaintArtwork }?.let {
            ProgressBarLine(progress = it)
        }
        outpaintStatus?.let {
            Text(
                text = it,
                modifier = Modifier.fillMaxWidth(),
                style = MaterialTheme.typography.labelSmall,
                color = if (it.contains("ready", ignoreCase = true)) Teal else TextSecondary,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
        OutlinedButton(
            onClick = {
                offsetX = 0f
                offsetY = 0f
                coverScale = 100f
                showTemplateOverlay = true
                extraLayerOffsetX = 0f
                extraLayerOffsetY = 0f
                extraLayerScale = 100f
                outpaintArtwork = false
                outpaintFill = null
                outpaintStatus = null
                outpaintProgress = null
            },
            modifier = Modifier.fillMaxWidth(),
            border = BorderStroke(1.dp, Blue)
        ) {
            Text("Reset Alignment")
        }
        selectedTemplate?.let { template ->
            if (template.supportsGameBoyLogoToggle || template.supports3dsSealToggle || template.supportsGameCodeGeneration) {
                HorizontalDivider(color = Blue.copy(alpha = 0.35f))
                if (template.supportsGameBoyLogoToggle) {
                    OverlayControls(
                        label = "Nintendo Logo",
                        enabled = showGameBoyLogo,
                        onEnabledChange = { showGameBoyLogo = it },
                        scale = logoScale,
                        onScaleChange = { logoScale = it.roundToInt().toFloat() },
                        offsetX = logoOffsetX,
                        onOffsetXChange = { logoOffsetX = it.roundToInt().toFloat() },
                        offsetY = logoOffsetY,
                        onOffsetYChange = { logoOffsetY = it.roundToInt().toFloat() }
                    )
                }

                if (template.supports3dsSealToggle) {
                    OverlayControls(
                        label = "Nintendo Seal",
                        enabled = show3dsSeal,
                        onEnabledChange = { show3dsSeal = it },
                        scale = sealScale,
                        onScaleChange = { sealScale = it.roundToInt().toFloat() },
                        offsetX = sealOffsetX,
                        onOffsetXChange = { sealOffsetX = it.roundToInt().toFloat() },
                        offsetY = sealOffsetY,
                        onOffsetYChange = { sealOffsetY = it.roundToInt().toFloat() }
                    )
                }

                if (template.supportsGameCodeGeneration) {
                    ToggleRow("Generate Game Code", generateGameCode) { generateGameCode = it }
                    if (generateGameCode) {
                        GameCodeControls(
                            titleQuery = gameTitleQuery,
                            onTitleQueryChange = {
                                gameTitleQuery = it
                                gameCodeLookupStatus = null
                            },
                            gameCode = gameCodeText,
                            onGameCodeChange = {
                                gameCodeText = it.uppercase(Locale.US).take(32)
                                gameCodeLookupStatus = null
                            },
                            bold = gameCodeBold,
                            onBoldChange = { gameCodeBold = it },
                            sizeStep = gameCodeSizeStep,
                            onDecreaseSize = { gameCodeSizeStep = (gameCodeSizeStep - 1).coerceAtLeast(-6) },
                            onIncreaseSize = { gameCodeSizeStep = (gameCodeSizeStep + 1).coerceAtMost(8) },
                            offsetX = gameCodeOffsetX,
                            onOffsetXChange = { gameCodeOffsetX = it.roundToInt().toFloat() },
                            offsetY = gameCodeOffsetY,
                            onOffsetYChange = { gameCodeOffsetY = it.roundToInt().toFloat() },
                            lookupStatus = gameCodeLookupStatus,
                            onFindCode = { gameCodeLookupRequest++ }
                        )
                    }
                }
            }
        }
    }

    val step4Section: @Composable () -> Unit = {
        StepLabel("4", "Output")
        PathLine("Folder", selectedTemplate?.defaultOutputDirectory()?.displayPath().orEmpty())
        Text(
            text = "Exports resized artwork only; template overlay is not included.",
            style = MaterialTheme.typography.labelSmall,
            color = TextSecondary,
            maxLines = 2,
            overflow = TextOverflow.Ellipsis
        )
        HorizontalDivider(color = Blue.copy(alpha = 0.55f))
        OutlinedButton(
            onClick = { saveAsLauncher.launch(makeExportName(selectedTemplate, selectedCover)) },
            modifier = Modifier.fillMaxWidth(),
            border = BorderStroke(1.dp, Blue)
        ) {
            Text("Choose Artwork Destination")
        }
        Button(
            onClick = {
                val bitmap = artworkExportBitmap
                val template = selectedTemplate
                if (bitmap == null || template == null) {
                    status = "Nothing to export"
                } else {
                    exportToDefaultFolder(context, bitmap, template, selectedCover)
                        .onSuccess { status = "Exported artwork ${it.name}" }
                        .onFailure { status = it.localizedMessage ?: "Export failed" }
                }
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Export Artwork PNG")
        }
        Text(
            text = status,
            modifier = Modifier.fillMaxWidth(),
            style = MaterialTheme.typography.labelMedium,
            color = Teal,
            textAlign = TextAlign.Center,
            maxLines = 2,
            overflow = TextOverflow.Ellipsis
        )
    }

    if (dualDisplayMode) {
        Row(
            modifier = Modifier
                .fillMaxSize()
                .background(AppBackground)
                .padding(horizontal = 18.dp, vertical = 12.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            ControlPanelCard(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxSize()
            ) {
                step1Section()
                step2Section()
            }
            ControlPanelCard(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxSize()
            ) {
                step3Section()
                step4Section()
            }
        }
    } else {
        Row(
            modifier = Modifier
                .fillMaxSize()
                .background(AppBackground)
                .padding(horizontal = 18.dp, vertical = 14.dp),
            horizontalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            ControlPanelCard(
                modifier = Modifier
                    .width(380.dp)
                    .fillMaxSize()
            ) {
                step1Section()
                step2Section()
                step3Section()
                step4Section()
            }
            PreviewPane(
                bitmap = previewBitmap,
                modifier = Modifier.fillMaxSize(),
                onTap = { dualDisplayMode = true },
                onBottomLayerGesture = { panX, panY, zoom ->
                    if (zoom.isFinite() && zoom > 0f) coverScale = (coverScale * zoom).coerceIn(5f, 220f)
                    if (panX.isFinite()) offsetX = (offsetX + panX).coerceIn(-500f, 500f)
                    if (panY.isFinite()) offsetY = (offsetY + panY).coerceIn(-500f, 500f)
                }
            )
        }
    }
}

@Composable
private fun CartridgeStamperTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(
            primary = HotPink,
            secondary = Teal,
            background = AppBackground,
            surface = PanelBackground,
            surfaceVariant = ControlBackground
        ),
        content = content
    )
}

private fun launchLowerDisplayPreview(context: Context, bitmap: Bitmap?): String? {
    val displayManager = context.getSystemService(Context.DISPLAY_SERVICE) as DisplayManager
    val defaultDisplayId = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
        context.display.displayId
    } else {
        android.view.Display.DEFAULT_DISPLAY
    }
    val presentationDisplay = displayManager
        .getDisplays(DisplayManager.DISPLAY_CATEGORY_PRESENTATION)
        .firstOrNull { it.displayId != defaultDisplayId }
    val targetDisplayId = presentationDisplay?.displayId
        ?: displayManager.displays.firstOrNull { it.displayId == 4 }?.displayId
        ?: displayManager.displays.firstOrNull { it.displayId != defaultDisplayId }?.displayId
        ?: return "No lower display found"

    val intent = Intent(context, LowerDisplayActivity::class.java).apply {
        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP)
        bitmap?.let { source ->
            writeLowerPreviewFile(context, source)?.let { filePath ->
                putExtra(EXTRA_PREVIEW_COMMAND, COMMAND_UPDATE_LOWER_PREVIEW)
                putExtra(EXTRA_BITMAP_FILE, filePath)
            }
        }
    }
    val options = ActivityOptions.makeBasic().apply {
        launchDisplayId = targetDisplayId
    }

    return runCatching {
        context.startActivity(intent, options.toBundle())
        null
    }.getOrElse {
        Log.e("CartridgeStamper", "Lower display launch failed", it)
        "Lower display launch failed"
    }
}

private fun closeLowerDisplayPreview(context: Context) {
    runCatching {
        context.sendBroadcast(
            Intent(ACTION_CLOSE_LOWER_DISPLAY).apply {
                setPackage(context.packageName)
            }
        )
    }
}

private fun sendLowerPreviewUpdate(context: Context, bitmap: Bitmap) {
    val filePath = writeLowerPreviewFile(context, bitmap) ?: return
    context.sendBroadcast(
        Intent(ACTION_LOWER_PREVIEW).apply {
            setPackage(context.packageName)
            putExtra(EXTRA_PREVIEW_COMMAND, COMMAND_UPDATE_LOWER_PREVIEW)
            putExtra(EXTRA_BITMAP_FILE, filePath)
        }
    )
}

private fun writeLowerPreviewFile(context: Context, bitmap: Bitmap): String? {
    return runCatching {
        val dir = File(context.cacheDir, "lower-display-preview").apply { mkdirs() }
        val file = File(dir, "preview-${System.nanoTime()}.png")
        FileOutputStream(file).use { output ->
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, output)
        }
        dir.listFiles()
            ?.sortedByDescending { it.lastModified() }
            ?.drop(8)
            ?.forEach { it.delete() }
        file.absolutePath
    }.onFailure {
        Log.e("CartridgeStamper", "Lower preview cache write failed", it)
    }.getOrNull()
}

@Composable
private fun ControlPanelCard(
    modifier: Modifier,
    content: @Composable () -> Unit
) {
    Card(
        modifier = modifier,
        shape = RoundedCornerShape(8.dp),
        colors = CardDefaults.cardColors(containerColor = PanelBackground)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            content()
        }
    }
}

@Composable
private fun PreviewPane(
    bitmap: Bitmap?,
    modifier: Modifier = Modifier,
    onTap: () -> Unit,
    onBottomLayerGesture: (Float, Float, Float) -> Unit
) {
    Box(
        modifier = modifier
            .pointerInput(bitmap) {
                detectTapGestures { onTap() }
            }
            .pointerInput(bitmap) {
                detectTransformGestures { _, pan, zoom, _ ->
                    val activeBitmap = bitmap ?: return@detectTransformGestures
                    val fitScale = min(
                        size.width.toFloat() / activeBitmap.width.toFloat(),
                        size.height.toFloat() / activeBitmap.height.toFloat()
                    ).coerceAtLeast(0.001f)
                    onBottomLayerGesture(
                        pan.x / fitScale,
                        -pan.y / fitScale,
                        zoom
                    )
                }
            },
        contentAlignment = Alignment.Center
    ) {
        WorkspaceGrid()
        if (bitmap != null) {
            Image(
                bitmap = bitmap.asImageBitmap(),
                contentDescription = "Stamped cartridge preview",
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Fit
            )
        } else {
            EmptyPreview()
        }
    }
}

@Composable
private fun WorkspaceGrid(modifier: Modifier = Modifier.fillMaxSize()) {
    Box(
        modifier = modifier
            .background(Color(0xFF2A2D31))
            .border(1.dp, Color(0xFF555B62).copy(alpha = 0.35f))
            .drawWorkspaceGrid()
    )
}

private fun Modifier.drawWorkspaceGrid(): Modifier = drawBehind {
    val minor = 24.dp.toPx()
    val minorColor = Color(0xFF3A3F45)
    val majorColor = Color(0xFF555B62)
    var x = 0f
    var xIndex = 0
    while (x <= size.width) {
        val color = if (xIndex % 4 == 0) majorColor else minorColor
        drawLine(color = color, start = androidx.compose.ui.geometry.Offset(x, 0f), end = androidx.compose.ui.geometry.Offset(x, size.height), strokeWidth = 1f)
        x += minor
        xIndex++
    }
    var y = 0f
    var yIndex = 0
    while (y <= size.height) {
        val color = if (yIndex % 4 == 0) majorColor else minorColor
        drawLine(color = color, start = androidx.compose.ui.geometry.Offset(0f, y), end = androidx.compose.ui.geometry.Offset(size.width, y), strokeWidth = 1f)
        y += minor
        yIndex++
    }
}

@Composable
private fun StepLabel(step: String, text: String) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Box(
            modifier = Modifier
                .size(24.dp)
                .background(HotPink, RoundedCornerShape(6.dp)),
            contentAlignment = Alignment.Center
        ) {
            Text(step, style = MaterialTheme.typography.labelMedium, color = TextPrimary)
        }
        Text(
            text = text,
            style = MaterialTheme.typography.titleSmall,
            color = TextPrimary,
            fontWeight = FontWeight.SemiBold
        )
    }
}

@Composable
private fun TemplateDropdown(
    templates: List<TemplateAsset>,
    selected: TemplateAsset?,
    onSelect: (TemplateAsset) -> Unit
) {
    var expanded by remember { mutableStateOf(false) }

    Box(modifier = Modifier.fillMaxWidth()) {
        OutlinedButton(
            onClick = { expanded = true },
            modifier = Modifier.fillMaxWidth(),
            border = BorderStroke(1.dp, Blue)
        ) {
            Column(modifier = Modifier.fillMaxWidth(), horizontalAlignment = Alignment.Start) {
                Text(
                    text = selected?.label ?: "No bundled templates found",
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                selected?.let {
                    Text(
                        text = "${it.systemName} / ${it.systemFolder}",
                        style = MaterialTheme.typography.labelSmall,
                        color = Teal,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
            }
        }
        ThemedDropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            Column(
                modifier = Modifier
                    .width(320.dp)
                    .heightIn(max = 620.dp)
                    .verticalScroll(rememberScrollState())
            ) {
                templates.forEach { template ->
                    DropdownMenuItem(
                        text = {
                            Column {
                                Text(template.label, color = TextPrimary)
                                Text(
                                    text = "${template.systemName} / ${template.systemFolder}",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = Teal
                                )
                            }
                        },
                        colors = MenuDefaults.itemColors(textColor = TextPrimary),
                        onClick = {
                            expanded = false
                            onSelect(template)
                        }
                    )
                }
            }
        }
    }
}

@Composable
private fun MediaFolderDropdown(
    selected: MediaFolderKind,
    onSelect: (MediaFolderKind) -> Unit
) {
    var expanded by remember { mutableStateOf(false) }

    Box(modifier = Modifier.fillMaxWidth()) {
        OutlinedButton(
            onClick = { expanded = true },
            modifier = Modifier.fillMaxWidth(),
            border = BorderStroke(1.dp, Blue)
        ) {
            Column(modifier = Modifier.fillMaxWidth(), horizontalAlignment = Alignment.Start) {
                Text(selected.label)
                Text(
                    text = selected.selectedDetail,
                    style = MaterialTheme.typography.labelSmall,
                    color = Teal
                )
            }
        }
        ThemedDropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            MediaFolderKind.entries.forEach { kind ->
                DropdownMenuItem(
                    text = {
                        Column {
                            Text(kind.label, color = TextPrimary)
                            Text(
                                text = kind.menuDetail,
                                style = MaterialTheme.typography.labelSmall,
                                color = Teal
                            )
                        }
                    },
                    colors = MenuDefaults.itemColors(textColor = TextPrimary),
                    onClick = {
                        expanded = false
                        onSelect(kind)
                    }
                )
            }
        }
    }
}

@Composable
private fun ThemedDropdownMenu(
    expanded: Boolean,
    onDismissRequest: () -> Unit,
    content: @Composable () -> Unit
) {
    DropdownMenu(
        expanded = expanded,
        onDismissRequest = onDismissRequest,
        shape = RoundedCornerShape(8.dp),
        containerColor = ControlBackground,
        tonalElevation = 0.dp,
        shadowElevation = 0.dp,
        border = BorderStroke(1.dp, Blue.copy(alpha = 0.9f)),
    ) {
        content()
    }
}

@Composable
private fun ToggleRow(
    label: String,
    checked: Boolean,
    modifier: Modifier = Modifier.fillMaxWidth(),
    onCheckedChange: (Boolean) -> Unit
) {
    Row(
        modifier = modifier,
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(label, style = MaterialTheme.typography.labelMedium, color = TextPrimary)
        Switch(checked = checked, onCheckedChange = onCheckedChange)
    }
}

@Composable
private fun OverlayControls(
    label: String,
    enabled: Boolean,
    onEnabledChange: (Boolean) -> Unit,
    scale: Float,
    onScaleChange: (Float) -> Unit,
    offsetX: Float,
    onOffsetXChange: (Float) -> Unit,
    offsetY: Float,
    onOffsetYChange: (Float) -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        ToggleRow(label, enabled, onCheckedChange = onEnabledChange)
        if (enabled) {
            ControlSlider("Scale", scale, 20f..220f, onScaleChange)
            ControlSlider("X", offsetX, -500f..500f, onOffsetXChange)
            ControlSlider("Y", offsetY, -500f..500f, onOffsetYChange)
        }
    }
}

@Composable
private fun GameCodeControls(
    titleQuery: String,
    onTitleQueryChange: (String) -> Unit,
    gameCode: String,
    onGameCodeChange: (String) -> Unit,
    bold: Boolean,
    onBoldChange: (Boolean) -> Unit,
    sizeStep: Int,
    onDecreaseSize: () -> Unit,
    onIncreaseSize: () -> Unit,
    offsetX: Float,
    onOffsetXChange: (Float) -> Unit,
    offsetY: Float,
    onOffsetYChange: (Float) -> Unit,
    lookupStatus: String?,
    onFindCode: () -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
        CompactOutlinedTextField(
            label = "Search Title",
            value = titleQuery,
            onValueChange = onTitleQueryChange
        )
        CompactOutlinedTextField(
            label = "Game Code",
            value = gameCode,
            onValueChange = onGameCodeChange
        )
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            ToggleRow(
                label = "Bold Code",
                checked = bold,
                onCheckedChange = onBoldChange,
                modifier = Modifier.weight(1f)
            )
            TextSymbolButton("-", onDecreaseSize)
            Text(
                text = sizeStep.signedStepLabel(),
                modifier = Modifier.width(32.dp),
                style = MaterialTheme.typography.labelMedium,
                color = TextPrimary,
                textAlign = TextAlign.Center
            )
            TextSymbolButton("+", onIncreaseSize)
        }
        ControlSlider("Code X", offsetX, -260f..260f, onOffsetXChange)
        ControlSlider("Code Height", offsetY, -220f..220f, onOffsetYChange)
        OutlinedButton(
            onClick = onFindCode,
            modifier = Modifier.fillMaxWidth(),
            border = BorderStroke(1.dp, Blue)
        ) {
            Text("Find Code")
        }
        lookupStatus?.let {
            Text(
                text = it,
                modifier = Modifier.fillMaxWidth(),
                style = MaterialTheme.typography.labelSmall,
                color = if (it.startsWith("Found")) Teal else TextSecondary,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

@Composable
private fun TextSymbolButton(symbol: String, onClick: () -> Unit) {
    Text(
        text = symbol,
        modifier = Modifier
            .clickable(onClick = onClick)
            .padding(horizontal = 10.dp, vertical = 2.dp),
        style = MaterialTheme.typography.titleMedium,
        color = Blue,
        fontWeight = FontWeight.ExtraBold,
        textAlign = TextAlign.Center
    )
}

private fun Int.signedStepLabel(): String {
    return when {
        this > 0 -> "+$this"
        else -> toString()
    }
}

@Composable
private fun CompactOutlinedTextField(
    label: String,
    value: String,
    onValueChange: (String) -> Unit
) {
    val focusManager = LocalFocusManager.current
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        modifier = Modifier.fillMaxWidth(),
        singleLine = true,
        keyboardOptions = KeyboardOptions(
            capitalization = KeyboardCapitalization.Characters,
            autoCorrectEnabled = false,
            keyboardType = KeyboardType.Ascii,
            imeAction = ImeAction.Done
        ),
        keyboardActions = KeyboardActions(
            onDone = { focusManager.clearFocus(force = true) }
        ),
        textStyle = MaterialTheme.typography.labelMedium.copy(color = TextPrimary),
        label = {
            Text(
                text = label,
                style = MaterialTheme.typography.labelSmall,
                color = TextSecondary
            )
        },
        colors = OutlinedTextFieldDefaults.colors(
            focusedBorderColor = Blue,
            unfocusedBorderColor = Blue,
            focusedTextColor = TextPrimary,
            unfocusedTextColor = TextPrimary,
            cursorColor = Teal,
            focusedContainerColor = ControlBackground,
            unfocusedContainerColor = ControlBackground
        ),
        shape = RoundedCornerShape(8.dp)
    )
}

@Composable
private fun PathLine(label: String, path: String) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(
            label,
            modifier = Modifier.width(64.dp),
            style = MaterialTheme.typography.labelMedium,
            color = TextSecondary
        )
        Text(
            path.ifBlank { "-" },
            modifier = Modifier.weight(1f),
            style = MaterialTheme.typography.labelMedium,
            color = TextPrimary,
            textAlign = TextAlign.End,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
    }
}

@Composable
private fun AssetList(
    assets: List<ImageAsset>,
    selected: ImageAsset?,
    emptyText: String,
    onSelect: (ImageAsset) -> Unit
) {
    if (assets.isEmpty()) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(60.dp)
                .border(1.dp, Blue, RoundedCornerShape(8.dp)),
            contentAlignment = Alignment.Center
        ) {
            Text(emptyText, style = MaterialTheme.typography.labelMedium, color = TextSecondary)
        }
        return
    }

    LazyColumn(
        modifier = Modifier
            .fillMaxWidth()
            .height(118.dp),
        verticalArrangement = Arrangement.spacedBy(5.dp)
    ) {
        items(assets, key = { it.source.key }) { asset ->
            AssetRow(asset = asset, selected = asset == selected, onClick = { onSelect(asset) })
        }
    }
}

@Composable
private fun AssetRow(asset: ImageAsset, selected: Boolean, onClick: () -> Unit) {
    val borderColor = Blue
    val background = if (selected) HotPink.copy(alpha = 0.12f) else ControlBackground
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .border(1.dp, borderColor, RoundedCornerShape(8.dp))
            .background(background, RoundedCornerShape(8.dp))
            .clickable(onClick = onClick)
            .padding(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .size(30.dp)
                .background(Blue.copy(alpha = 0.16f), RoundedCornerShape(6.dp)),
            contentAlignment = Alignment.Center
        ) {
            Text(asset.extension.uppercase(Locale.US).take(3), style = MaterialTheme.typography.labelSmall, color = Teal)
        }
        Spacer(Modifier.width(8.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(asset.label, style = MaterialTheme.typography.labelMedium, color = TextPrimary, maxLines = 1, overflow = TextOverflow.Ellipsis)
            Text(asset.detail, style = MaterialTheme.typography.labelSmall, color = TextSecondary, maxLines = 1, overflow = TextOverflow.Ellipsis)
        }
    }
}

@Composable
private fun ControlSlider(label: String, value: Float, range: ClosedFloatingPointRange<Float>, onChange: (Float) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(label, style = MaterialTheme.typography.labelSmall, color = TextSecondary)
            Text(value.toInt().toString(), style = MaterialTheme.typography.labelSmall, color = TextPrimary)
        }
        CompactSlider(value = value, range = range, onChange = onChange)
    }
}

@Composable
private fun CompactSlider(value: Float, range: ClosedFloatingPointRange<Float>, onChange: (Float) -> Unit) {
    val clampedValue = value.coerceIn(range.start, range.endInclusive)
    val span = (range.endInclusive - range.start).coerceAtLeast(1f)
    val fraction = ((clampedValue - range.start) / span).coerceIn(0f, 1f)
    val thumbSize = 10.dp

    fun valueFromX(x: Float, width: Int): Float {
        val localFraction = (x / width.coerceAtLeast(1).toFloat()).coerceIn(0f, 1f)
        return range.start + span * localFraction
    }

    BoxWithConstraints(
        modifier = Modifier
            .fillMaxWidth()
            .height(24.dp)
            .pointerInput(range) {
                detectTapGestures { offset ->
                    onChange(valueFromX(offset.x, size.width))
                }
            }
            .pointerInput(range) {
                detectDragGestures { change, _ ->
                    onChange(valueFromX(change.position.x, size.width))
                    change.consume()
                }
            },
        contentAlignment = Alignment.CenterStart
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(3.dp)
                .background(TrackBackground, RoundedCornerShape(50))
        )
        Box(
            modifier = Modifier
                .fillMaxWidth(fraction)
                .height(3.dp)
                .background(Blue, RoundedCornerShape(50))
        )
        val thumbOffset: Dp = (maxWidth - thumbSize) * fraction
        Box(
            modifier = Modifier
                .offset(x = thumbOffset)
                .size(thumbSize)
                .background(Blue, RoundedCornerShape(50)),
        )
    }
}

@Composable
private fun ProgressBarLine(progress: OutpaintProgress) {
    val fraction = progress.fraction?.coerceIn(0f, 1f) ?: 0f
    Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(progress.label, style = MaterialTheme.typography.labelSmall, color = TextSecondary, maxLines = 1, overflow = TextOverflow.Ellipsis)
            progress.fraction?.let {
                Text("${(it * 100f).roundToInt()}%", style = MaterialTheme.typography.labelSmall, color = Teal)
            }
        }
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(5.dp)
                .background(TrackBackground, RoundedCornerShape(50))
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth(fraction.coerceAtLeast(0.03f))
                    .height(5.dp)
                    .background(HotPink, RoundedCornerShape(50))
            )
        }
    }
}

@Composable
private fun EmptyPreview() {
    Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text("No preview yet", style = MaterialTheme.typography.titleMedium, color = TextPrimary)
        Text("Select a bundled template", style = MaterialTheme.typography.labelMedium, color = TextSecondary)
    }
}

private enum class MediaFolderKind(val label: String, val directoryName: String?) {
    PhysicalMedia("Physical Media", "physicalmedia"),
    Covers("Covers", "covers"),
    SpecificImage("Specific Image", null),
    CustomFolder("Specific Folder", null);

    val isDefaultSource: Boolean
        get() = directoryName != null

    val menuDetail: String
        get() = when (this) {
            SpecificImage -> "Choose one artwork file"
            CustomFolder -> "Choose a specific folder"
            else -> "downloaded_media/<system>/$directoryName"
        }

    val selectedDetail: String
        get() = when (this) {
            SpecificImage -> "Single image"
            CustomFolder -> "Custom folder"
            else -> "ES-DE / $directoryName"
        }
}

private data class TemplateAsset(
    val label: String,
    val assetPath: String,
    val systemName: String,
    val systemFolder: String,
    val templateId: String,
    val maskPath: String? = null
) {
    fun defaultMediaDirectory(mediaFolderKind: MediaFolderKind): File? = esDeMediaDirectory(systemFolder, mediaFolderKind)

    fun defaultOutputDirectory(): File = File(defaultOutputRootDirectory(), systemFolder)

    val supportsGameBoyLogoToggle: Boolean
        get() = systemFolder == "gb"

    val supports3dsSealToggle: Boolean
        get() = systemFolder == "n3ds" || systemFolder == "nds"

    val supportsGameCodeGeneration: Boolean
        get() = (systemFolder == "n3ds" && (templateId.contains("sticker", ignoreCase = true) || label.contains("w/ Code", ignoreCase = true))) ||
            systemFolder == "nds" ||
            systemFolder == "switch"
}

private data class ImageAsset(
    val label: String,
    val detail: String,
    val source: ImageSource
) {
    val extension: String = label.substringAfterLast('.', missingDelimiterValue = "img")
}

private data class OutpaintFill(
    val key: String,
    val bitmap: Bitmap
)

private data class OutpaintProgress(
    val label: String,
    val fraction: Float?
)

private sealed interface ImageSource {
    val key: String

    data class FileSource(val file: File) : ImageSource {
        override val key: String = file.absolutePath
    }

    data class UriSource(val uri: Uri) : ImageSource {
        override val key: String = uri.toString()
    }
}

private data class GameTdbSearchRow(
    val id: String,
    val description: String
)

private suspend fun queryGameProductCode(
    titleQuery: String?,
    template: TemplateAsset,
    originalLabel: String?
): String? = withContext(Dispatchers.IO) {
    val label = originalLabel.orEmpty()
    val title = titleQuery.orEmpty().cleanGameTitleForLookup()
    val regionSource = "$label $title"
    val regionSuffix = regionSource.detectRegionSuffix()
    val platform = template.gameTdbPlatform() ?: return@withContext null
    if (title.isBlank()) return@withContext null

    extractProductCodeFromText(regionSource, template)?.let {
        return@withContext it
    }

    knownGameTdbId(template, title, regionSource)?.let {
        return@withContext template.formatGameProductCode(it, regionSuffix)
    }

    val gameTdbCode = runCatching {
        val encodedTitle = URLEncoder.encode(title, "UTF-8")
        val encodedGroup = URLEncoder.encode("group=$platform", "UTF-8")
        val url = URL("https://www.gametdb.com/$platform/Search?action=search&q=$encodedGroup&title_EN=$encodedTitle")
        val connection = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 5000
            readTimeout = 5000
            setRequestProperty("User-Agent", "CartridgeStamper/0.1")
        }
        connection.inputStream.bufferedReader().use { it.readText() }
            .parseGameTdbId(platform, title, regionSource)
            ?.let { template.formatGameProductCode(it, regionSuffix) }
    }.getOrNull()
    gameTdbCode ?: if (template.systemFolder == "switch") {
        querySwitchSerialFromMarioWiki(title, regionSuffix)
    } else {
        null
    }
}

private fun TemplateAsset.gameTdbPlatform(): String? {
    return when (systemFolder) {
        "n3ds" -> "3DS"
        "nds" -> "DS"
        "switch" -> "Switch"
        else -> null
    }
}

private fun String.cleanGameTitleForLookup(): String {
    return substringBeforeLast('.', this)
        .replace(Regex("""\[[^\]]*]"""), " ")
        .replace(Regex("""\([^)]*\)"""), " ")
        .replace(Regex("""(?i)\b(usa|eur|europe|jpn|japan|pal|ntsc-u|ntsc-j|rev\s*\d+|v\d+)\b"""), " ")
        .replace(Regex("""[_+.-]+"""), " ")
        .replace(Regex("""\s+"""), " ")
        .trim()
}

private fun String.normalizeManualGameCode(): String {
    return uppercase(Locale.US)
        .replace(Regex("""[^A-Z0-9-]"""), "")
        .take(32)
}

private fun String.detectRegionSuffix(): String {
    val lower = lowercase(Locale.US)
    return when {
        "eur" in lower || "europe" in lower || "pal" in lower -> "EUR"
        "jpn" in lower || "japan" in lower || "ntsc-j" in lower -> "JPN"
        else -> "USA"
    }
}

private fun extractProductCodeFromText(text: String, template: TemplateAsset): String? {
    val upper = text.uppercase(Locale.US)
    val fullPattern = when (template.systemFolder) {
        "n3ds" -> Regex("""LNA-CTR-[A-Z0-9]{3,8}-(USA|EUR|JPN)""")
        "nds" -> Regex("""NTR-[A-Z0-9]{3,8}-(USA|EUR|JPN)""")
        "switch" -> Regex("""LA-H-[A-Z0-9]{5}-(USA|EUR|JPN)""")
        else -> return null
    }
    fullPattern.find(upper)?.let { return it.value }

    val region = upper.detectRegionSuffix()
    val shortPattern = when (template.systemFolder) {
        "n3ds" -> Regex("""\bCTR[-_ ]?([A-Z0-9]{3,8})\b""")
        "nds" -> Regex("""\bNTR[-_ ]?([A-Z0-9]{3,8})\b""")
        "switch" -> Regex("""\bHAC[-_ ]?([A-Z0-9]{5})\b""")
        else -> return null
    }
    return shortPattern.find(upper)?.groupValues?.getOrNull(1)?.let {
        template.formatGameProductCode(it, region)
    }
}

private fun knownGameTdbId(template: TemplateAsset, title: String, label: String): String? {
    val normalized = title.lowercase(Locale.US)
    val region = label.detectRegionSuffix()
    return when (template.systemFolder) {
        "n3ds" -> when {
            normalized == "angry birds star wars" && region == "EUR" -> "ANDP"
            normalized == "angry birds star wars" -> "ANDE"
            else -> null
        }

        "switch" -> when (normalized.normalizedLookupKey()) {
            "mario golf super rush" -> "AT9HA"
            "super mario odyssey" -> "AAACA"
            else -> null
        }

        else -> null
    }
}

private fun querySwitchSerialFromMarioWiki(title: String, regionSuffix: String): String? {
    return runCatching {
        val page = title.cleanGameTitleForLookup()
            .replace(":", "")
            .replace(Regex("""\s+"""), "_")
        val url = URL("https://www.mariowiki.com/${URLEncoder.encode(page, "UTF-8")}")
        val connection = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 5000
            readTimeout = 5000
            setRequestProperty("User-Agent", "CartridgeStamper/0.1")
        }
        val body = connection.inputStream.bufferedReader().use { it.readText() }
        val id = Regex("""HAC-([A-Z0-9]{5})-[A-Z]{3}""", RegexOption.IGNORE_CASE)
            .find(body)
            ?.groupValues
            ?.getOrNull(1)
            ?.uppercase(Locale.US)
        id?.let { "LA-H-$it-$regionSuffix" }
    }.getOrNull()
}

private fun String.parseGameTdbId(platform: String, titleQuery: String, originalLabel: String): String? {
    val escapedPlatform = Regex.escape(platform)
    val rows = Regex(
        """<td[^>]*>\s*<a[^>]*href=['"]https://www\.gametdb\.com/$escapedPlatform/([A-Z0-9]{3,16})['"][^>]*>[^<]*</a>\s*</td>\s*<td[^>]*>\s*([^<]+)""",
        RegexOption.IGNORE_CASE
    ).findAll(this).map {
        GameTdbSearchRow(
            id = it.groupValues[1].uppercase(Locale.US),
            description = it.groupValues[2].stripHtmlEntities().trim()
        )
    }.toList()
    if (rows.isEmpty()) return null

    val region = originalLabel.detectRegionSuffix()
    return rows.maxByOrNull { it.scoreFor(titleQuery, region) }?.id
}

private fun GameTdbSearchRow.scoreFor(titleQuery: String, region: String): Int {
    val queryTitle = titleQuery.cleanGameTitleForLookup()
    val rowTitle = description.substringBeforeLast(" (", description).trim()
    val queryNormalized = queryTitle.normalizedLookupKey()
    val rowNormalized = rowTitle.cleanGameTitleForLookup().normalizedLookupKey()
    val regionNeedle = when (region) {
        "EUR" -> "PAL"
        "JPN" -> "NTSC-J"
        else -> "NTSC-U"
    }

    var score = 0
    if (rowTitle.equals(queryTitle, ignoreCase = true)) score += 90
    if (rowNormalized == queryNormalized && rowNormalized.isNotBlank()) score += 70
    if (rowNormalized.contains(queryNormalized) || queryNormalized.contains(rowNormalized)) score += 20
    if (description.contains(regionNeedle, ignoreCase = true)) score += 32
    if (description.contains("ALL", ignoreCase = true)) score += 18
    if (description.contains(region, ignoreCase = true)) score += 24
    if (description.contains("Demo", ignoreCase = true)) score -= 40
    if (description.contains("Kiosk", ignoreCase = true)) score -= 40
    if (description.contains("Starter Pack", ignoreCase = true)) score -= 30
    if (rowNormalized.isBlank()) score -= 100
    return score
}

private fun String.normalizedLookupKey(): String {
    return lowercase(Locale.US)
        .replace(Regex("""[^a-z0-9]+"""), " ")
        .replace(Regex("""\s+"""), " ")
        .trim()
}

private fun String.stripHtmlEntities(): String {
    return replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#039;", "'")
        .replace("&nbsp;", " ")
}

private fun TemplateAsset.formatGameProductCode(gameId: String, regionSuffix: String): String {
    val normalizedId = gameId.uppercase(Locale.US)
    return when (systemFolder) {
        "n3ds" -> "LNA-CTR-$normalizedId-$regionSuffix"
        "nds" -> "NTR-$normalizedId-$regionSuffix"
        "switch" -> "LA-H-$normalizedId-$regionSuffix"
        else -> normalizedId
    }
}

private fun loadTemplateAssets(context: Context): List<TemplateAsset> {
    val entries = context.assets.list(TEMPLATE_ASSET_DIR)
        .orEmpty()
        .asSequence()

    return entries
        .flatMap { entry ->
            val entryPath = "$TEMPLATE_ASSET_DIR/$entry"
            val children = runCatching { context.assets.list(entryPath).orEmpty().toSet() }.getOrDefault(emptySet())
            when {
                "overlay.png" in children -> {
                    val metadata = templateMetadata(entry)
                    sequenceOf(
                        TemplateAsset(
                            label = metadata.label,
                            assetPath = "$entryPath/overlay.png",
                            systemName = metadata.systemName,
                            systemFolder = metadata.systemFolder,
                            templateId = entry,
                            maskPath = "$entryPath/mask.png".takeIf { "mask.png" in children }
                        )
                    )
                }

                entry.isSupportedImageName() &&
                    !entry.contains("Official Nintendo Seal", ignoreCase = true) &&
                    !entry.contains("Nintendo_Official_Seal", ignoreCase = true) &&
                    !entry.contains("nintendo-logo-vector", ignoreCase = true) -> {
                    val metadata = templateMetadata(entry)
                    sequenceOf(
                        TemplateAsset(
                            label = metadata.label,
                            assetPath = entryPath,
                            systemName = metadata.systemName,
                            systemFolder = metadata.systemFolder,
                            templateId = entry.removeImageExtension().sanitizeFilePart()
                        )
                    )
                }

                else -> emptySequence()
            }
        }
        .sortedWith(compareBy({ it.label.lowercase(Locale.US) }, { it.templateId.lowercase(Locale.US) }))
        .toList()
}

private fun appVersionLabel(context: Context): String {
    return runCatching {
        val versionName = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            context.packageManager.getPackageInfo(
                context.packageName,
                android.content.pm.PackageManager.PackageInfoFlags.of(0)
            ).versionName
        } else {
            @Suppress("DEPRECATION")
            context.packageManager.getPackageInfo(context.packageName, 0).versionName
        }
        "v.${versionName ?: APP_VERSION_FALLBACK.removePrefix("v.")}"
    }.getOrDefault(APP_VERSION_FALLBACK)
}

private data class TemplateMetadata(
    val label: String,
    val systemName: String,
    val systemFolder: String
)

private data class TemplateSystemRule(
    val id: String,
    val systemFolder: String,
    val baseLabel: String,
    val systemName: String
) {
    fun matches(identifier: String): Boolean {
        return identifier == id ||
            identifier.startsWith("${id}_") ||
            identifier.startsWith("${id}-")
    }
}

private val TemplateSystemRules = listOf(
    TemplateSystemRule("amiga1200", "amiga1200", "Amiga 1200", "Commodore Amiga 1200"),
    TemplateSystemRule("amiga600", "amiga600", "Amiga 600", "Commodore Amiga 600"),
    TemplateSystemRule("amiga", "amiga", "Amiga", "Commodore Amiga"),
    TemplateSystemRule("amstradcpc", "amstradcpc", "Amstrad CPC", "Amstrad CPC"),
    TemplateSystemRule("arcade", "arcade", "Arcade", "Arcade"),
    TemplateSystemRule("atomiswave", "atomiswave", "Atomiswave", "Sammy Atomiswave"),
    TemplateSystemRule("cps1", "cps1", "Capcom Play System", "Capcom Play System"),
    TemplateSystemRule("cps2", "cps2", "Capcom Play System II", "Capcom Play System II"),
    TemplateSystemRule("cps3", "cps3", "Capcom Play System III", "Capcom Play System III"),
    TemplateSystemRule("dos", "dos", "DOS", "DOS"),
    TemplateSystemRule("dreamcast", "dreamcast", "Dreamcast", "Sega Dreamcast"),
    TemplateSystemRule("fba", "fba", "FinalBurn Alpha", "FinalBurn Alpha"),
    TemplateSystemRule("fbneo", "fbneo", "FinalBurn Neo", "FinalBurn Neo"),
    TemplateSystemRule("fds", "fds", "Famicom Disk System", "Famicom Disk System"),
    TemplateSystemRule("gamegear", "gamegear", "Game Gear", "Sega Game Gear"),
    TemplateSystemRule("gba", "gba", "Game Boy Advance", "Game Boy Advance"),
    TemplateSystemRule("gbc", "gbc", "Game Boy Color", "Game Boy Color"),
    TemplateSystemRule("gb", "gb", "Game Boy", "Game Boy"),
    TemplateSystemRule("gc", "gc", "Nintendo GameCube", "Nintendo GameCube"),
    TemplateSystemRule("generic_dvd", "generic_dvd", "Generic DVD", "Generic DVD"),
    TemplateSystemRule("genesis", "genesis", "Sega Genesis", "Sega Genesis"),
    TemplateSystemRule("mame", "mame", "MAME", "MAME"),
    TemplateSystemRule("mastersystem", "mastersystem", "Master System", "Sega Master System"),
    TemplateSystemRule("megacd", "megacd", "Mega-CD", "Sega Mega-CD"),
    TemplateSystemRule("msx", "msx", "MSX", "MSX"),
    TemplateSystemRule("n3ds", "n3ds", "Nintendo 3DS", "Nintendo 3DS"),
    TemplateSystemRule("n64", "n64", "Nintendo 64", "Nintendo 64"),
    TemplateSystemRule("ndsi", "nds", "Nintendo DSi", "Nintendo DS"),
    TemplateSystemRule("nds", "nds", "Nintendo DS", "Nintendo DS"),
    TemplateSystemRule("nes", "nes", "NES", "Nintendo Entertainment System"),
    TemplateSystemRule("pc", "pc", "PC", "PC"),
    TemplateSystemRule("pico8", "pico8", "PICO-8", "PICO-8"),
    TemplateSystemRule("pokemini", "pokemini", "Pokemon Mini", "Pokemon Mini"),
    TemplateSystemRule("ps2", "ps2", "PlayStation 2", "Sony PlayStation 2"),
    TemplateSystemRule("ps3", "ps3", "PlayStation 3", "Sony PlayStation 3"),
    TemplateSystemRule("pspminis", "pspminis", "PlayStation Portable Minis", "Sony PlayStation Portable Minis"),
    TemplateSystemRule("psp", "psp", "PlayStation Portable UMD", "Sony PlayStation Portable"),
    TemplateSystemRule("psvita", "psvita", "PlayStation Vita", "Sony PlayStation Vita"),
    TemplateSystemRule("psx", "psx", "PlayStation", "Sony PlayStation"),
    TemplateSystemRule("saturn", "saturn", "Saturn", "Sega Saturn"),
    TemplateSystemRule("sega32x", "sega32x", "Sega 32X", "Sega 32X"),
    TemplateSystemRule("segacd", "segacd", "Sega CD", "Sega CD"),
    TemplateSystemRule("sg-1000", "sg1000", "SG-1000", "Sega SG-1000"),
    TemplateSystemRule("sg1000", "sg1000", "SG-1000", "Sega SG-1000"),
    TemplateSystemRule("snesmsu1", "snesmsu1", "Super Nintendo MSU-1", "Super Nintendo MSU-1"),
    TemplateSystemRule("snes", "snes", "Super Nintendo", "Super Nintendo"),
    TemplateSystemRule("steam_screen", "steam", "Steam Screen", "Steam"),
    TemplateSystemRule("steam", "steam", "Steam", "Steam"),
    TemplateSystemRule("supergrafx", "supergrafx", "SuperGrafx", "PC Engine SuperGrafx"),
    TemplateSystemRule("switch", "switch", "Nintendo Switch", "Nintendo Switch"),
    TemplateSystemRule("tg16", "tg16", "TurboGrafx-16", "TurboGrafx-16"),
    TemplateSystemRule("virtualboy", "virtualboy", "Virtual Boy", "Nintendo Virtual Boy"),
    TemplateSystemRule("wiiu", "wiiu", "Wii U", "Nintendo Wii U"),
    TemplateSystemRule("wiiware", "wiiware", "WiiWare", "Nintendo WiiWare"),
    TemplateSystemRule("wii", "wii", "Wii", "Nintendo Wii"),
    TemplateSystemRule("windows3x", "windows3x", "Windows 3.x", "Windows 3.x"),
    TemplateSystemRule("windows9x", "windows9x", "Windows 9x", "Windows 9x"),
    TemplateSystemRule("windows", "windows", "Windows", "Windows")
)

private fun templateMetadata(templateName: String): TemplateMetadata {
    val normalized = templateName.removeImageExtension().templateIdentifier()
    TemplateSystemRules.firstOrNull { it.matches(normalized) }?.let { rule ->
        val variant = normalized.removePrefix(rule.id).trimStart('_', '-').templateVariantLabel()
        return TemplateMetadata(
            label = if (variant.isBlank()) rule.baseLabel else "${rule.baseLabel} ($variant)",
            systemName = rule.systemName,
            systemFolder = rule.systemFolder
        )
    }

    val lower = templateName.lowercase(Locale.US)
    return when {
        "advance" in lower || Regex("""\bgba\b""").containsMatchIn(lower) ->
            TemplateMetadata("Game Boy Advance", "Game Boy Advance", "gba")
        "color" in lower || Regex("""\bgbc\b""").containsMatchIn(lower) ->
            TemplateMetadata("Game Boy Color", "Game Boy Color", "gbc")
        "gameboy" in lower || "game boy" in lower || Regex("""\bgb\b""").containsMatchIn(lower) ->
            TemplateMetadata("Game Boy", "Game Boy", "gb")
        "3ds" in lower && "code" in lower ->
            TemplateMetadata("Nintendo 3DS (w/ Code)", "Nintendo 3DS", "n3ds")
        "3ds" in lower ->
            TemplateMetadata("Nintendo 3DS", "Nintendo 3DS", "n3ds")
        "nds" in lower || "ds " in lower ->
            TemplateMetadata("Nintendo DS", "Nintendo DS", "nds")
        "super nintendo" in lower || "snes" in lower || Regex("""\bsfc\b""").containsMatchIn(lower) ->
            TemplateMetadata("Super Nintendo", "Super Nintendo", "snes")
        "genesis" in lower || "mega drive" in lower || "megadrive" in lower ->
            TemplateMetadata("Sega Genesis", "Sega Genesis", "genesis")
        "nes" in lower && "gold" in lower ->
            TemplateMetadata("NES (Gold)", "Nintendo Entertainment System", "nes")
        Regex("""(^|[^a-z0-9])nes([^a-z0-9]|$)""").containsMatchIn(lower) || "nintendo entertainment system" in lower ->
            TemplateMetadata("NES", "Nintendo Entertainment System", "nes")
        "gamecube" in lower || "game cube" in lower ->
            TemplateMetadata("Nintendo GameCube", "Nintendo GameCube", "gc")
        "psx" in lower || "playstation 1" in lower || Regex("""\bps1\b""").containsMatchIn(lower) ->
            TemplateMetadata("PlayStation", "Sony PlayStation", "psx")
        "ps2" in lower || "playstation 2" in lower ->
            TemplateMetadata("PlayStation 2", "Sony PlayStation 2", "ps2")
        "ps3" in lower || "playstation 3" in lower ->
            TemplateMetadata("PlayStation 3", "Sony PlayStation 3", "ps3")
        "psp" in lower || "umd" in lower ->
            TemplateMetadata("PlayStation Portable UMD", "Sony PlayStation Portable", "psp")
        "switch" in lower ->
            TemplateMetadata("Nintendo Switch", "Nintendo Switch", "switch")
        else ->
            TemplateMetadata(templateName.removeImageExtension().normalizeTemplateName().removeCartridgeWord(), "Custom", templateName.removeImageExtension().sanitizeFilePart())
    }
}

private fun String.templateIdentifier(): String {
    return lowercase(Locale.US)
        .replace(Regex("""\s+"""), "_")
        .replace(Regex("""[^a-z0-9_-]+"""), "_")
        .trim('_', '-')
}

private fun String.templateVariantLabel(): String {
    return split('_', '-')
        .filter { it.isNotBlank() }
        .map { it.templateVariantTokenLabel() }
        .joinToString(" ")
        .normalizeTemplateName()
}

private fun String.templateVariantTokenLabel(): String {
    return when (lowercase(Locale.US)) {
        "alt" -> "Alt"
        "black" -> "Black"
        "blacktext" -> "Black Text"
        "blue" -> "Blue"
        "cd" -> "CD"
        "crystal" -> "Crystal"
        "darkblue" -> "Dark Blue"
        "dvd" -> "DVD"
        "ea" -> "EA"
        "emerald" -> "Emerald"
        "famicom" -> "Famicom"
        "firered" -> "FireRed"
        "gold" -> "Gold"
        "gray" -> "Gray"
        "green" -> "Green"
        "jap" -> "Japan"
        "leafgreen" -> "LeafGreen"
        "lesssticker" -> "Less Sticker"
        "msu1" -> "MSU-1"
        "nes" -> "NES"
        "new3ds" -> "New 3DS"
        "new3dsblack" -> "New 3DS Black"
        "nocenter" -> "No Center"
        "nosticker" -> "No Sticker"
        "orangewhite" -> "Orange White"
        "orange" -> "Orange"
        "pink" -> "Pink"
        "purple" -> "Purple"
        "red" -> "Red"
        "ruby" -> "Ruby"
        "rumble" -> "Rumble"
        "salmon" -> "Salmon"
        "sapphire" -> "Sapphire"
        "screen" -> "Screen"
        "silver" -> "Silver"
        "sticker" -> "Sticker"
        "transparent" -> "Transparent"
        "usa" -> "USA"
        "white" -> "White"
        "whitelogo" -> "White Logo"
        "whitetext" -> "White Text"
        "yellow" -> "Yellow"
        else -> replaceFirstChar { if (it.isLowerCase()) it.titlecase(Locale.US) else it.toString() }
    }
}

private fun String.normalizeTemplateName(): String {
    return replace(Regex("""\s+"""), " ")
        .trim()
}

private fun String.removeImageExtension(): String = substringBeforeLast('.', this)

private fun String.removeCartridgeWord(): String {
    return replace(Regex("""\bcartridge\b""", RegexOption.IGNORE_CASE), "")
        .normalizeTemplateName()
}

private fun esDeMediaDirectory(systemFolder: String, mediaFolderKind: MediaFolderKind): File? {
    val directoryName = mediaFolderKind.directoryName ?: return null
    val underscored = File(Environment.getExternalStorageDirectory(), "ES-DE/downloaded_media/$systemFolder/$directoryName")
    if (underscored.exists()) return underscored
    val hyphenated = File(Environment.getExternalStorageDirectory(), "ES-DE/downloaded-media/$systemFolder/$directoryName")
    return if (hyphenated.exists()) hyphenated else underscored
}

private fun defaultOutputRootDirectory(): File {
    return File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "ICONS")
}

private fun ensureOutputFolders() {
    defaultOutputRootDirectory().mkdirs()
}

private fun File.displayPath(): String {
    val root = Environment.getExternalStorageDirectory().absolutePath
    return absolutePath.replace(root, "/AYN Thor")
}

private fun loadImagesFromFolder(folder: File): List<ImageAsset> {
    return folder.listFiles()
        ?.asSequence()
        ?.filter { it.isFile && it.name.isSupportedImageName() }
        ?.sortedBy { it.name.lowercase(Locale.US) }
        ?.map {
            ImageAsset(
                label = it.name,
                detail = it.parentFile?.displayPath().orEmpty(),
                source = ImageSource.FileSource(it)
            )
        }
        ?.toList()
        .orEmpty()
}

private fun loadImagesFromTree(context: Context, treeUri: Uri): List<ImageAsset> {
    val root = DocumentFile.fromTreeUri(context, treeUri) ?: return emptyList()
    return root.listFiles()
        .asSequence()
        .filter { it.isFile && (it.type?.startsWith("image/") == true || it.name?.isSupportedImageName() == true) }
        .sortedBy { it.name?.lowercase(Locale.US).orEmpty() }
        .map {
            ImageAsset(
                label = it.name ?: "Cover art",
                detail = "selected folder",
                source = ImageSource.UriSource(it.uri)
            )
        }
        .toList()
}

private fun imageAssetFromUri(context: Context, uri: Uri): ImageAsset {
    return ImageAsset(
        label = queryDisplayName(context, uri) ?: uri.lastPathSegment?.substringAfterLast('/') ?: "Cover art",
        detail = "selected image",
        source = ImageSource.UriSource(uri)
    )
}

private fun String.isSupportedImageName(): Boolean {
    val lower = lowercase(Locale.US)
    return lower.endsWith(".png") || lower.endsWith(".jpg") || lower.endsWith(".jpeg") || lower.endsWith(".webp")
}


private val TemplateAsset.isGameCubeTemplate: Boolean
    get() = systemFolder == "gc" ||
        systemFolder.equals("gamecube", ignoreCase = true) ||
        label.contains("gamecube", ignoreCase = true) ||
        assetPath.contains("gamecube", ignoreCase = true)

private val TemplateAsset.isDiscTemplate: Boolean
    get() = systemFolder in setOf("gc", "psp", "psx", "ps2", "ps3") ||
        label.contains("disc", ignoreCase = true) ||
        label.contains("umd", ignoreCase = true) ||
        label.contains("gamecube", ignoreCase = true)


private fun decodeTemplateBitmap(context: Context, template: TemplateAsset): Bitmap? {
    return runCatching {
        val options = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        context.assets.open(template.assetPath).use {
            BitmapFactory.decodeStream(it, null, options)
        }
    }.getOrNull()
}

private fun decodeTemplateMask(context: Context, template: TemplateAsset): Bitmap? {
    val maskPath = template.maskPath ?: return null
    return runCatching {
        val options = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        context.assets.open(maskPath).use {
            BitmapFactory.decodeStream(it, null, options)
        }
    }.getOrNull()
}


private val TemplateAsset.usesNintendoSeal: Boolean
    get() = systemFolder == "n3ds" || systemFolder == "nds"

private val TemplateAsset.supportsRatingToggle: Boolean
    get() = systemFolder == "nds"

private fun TemplateAsset.artworkWindow(context: Context, width: Int, height: Int): RectF? {
    decodeTemplateMask(context, this)?.let { mask ->
        val bounds = mask.alphaBounds()
        if (bounds != null) {
            val scaleX = width.toFloat() / mask.width.toFloat()
            val scaleY = height.toFloat() / mask.height.toFloat()
            return RectF(
                bounds.left * scaleX,
                bounds.top * scaleY,
                bounds.right * scaleX,
                bounds.bottom * scaleY
            )
        }
    }

    fun rect(left: Float, top: Float, right: Float, bottom: Float) = RectF(
        width * left,
        height * top,
        width * right,
        height * bottom
    )

    return when (systemFolder) {
        "n3ds" -> rect(
            left = 0.116f,
            top = if (label.contains("w/ Code", ignoreCase = true)) 0.212f else 0.210f,
            right = 0.884f,
            bottom = if (label.contains("w/ Code", ignoreCase = true)) 0.800f else 0.842f
        )

        "nds" -> rect(0.145f, 0.184f, 0.875f, 0.790f)
        "switch" -> rect(0.132f, 0.222f, 0.868f, 0.720f)
        "gb" -> rect(0.145f, 0.370f, 0.855f, 0.872f)
        "gbc" -> rect(0.145f, 0.315f, 0.855f, 0.850f)
        "gba" -> rect(0.115f, 0.285f, 0.885f, 0.770f)
        "gc" -> rect(0.190f, 0.190f, 0.810f, 0.810f)
        "nes" -> rect(0.396f, 0.002f, 0.875f, 0.682f)
        "snes" -> rect(0.275f, 0.026f, 0.610f, 0.380f)
        "psp" -> rect(0.155f, 0.245f, 0.845f, 0.755f)
        "psx", "ps2", "ps3" -> rect(0.018f, 0.018f, 0.982f, 0.982f)
        "genesis" -> rect(0.075f, 0.085f, 0.925f, 0.715f)
        else -> rect(0.140f, 0.220f, 0.860f, 0.820f)
    }
}
































private fun clearCenterWindowMarkings(bitmap: Bitmap, template: TemplateAsset) {
    if (template.systemFolder != "nds" && template.systemFolder != "n3ds") return

    val canvas = Canvas(bitmap)
    val clearPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        xfermode = PorterDuffXfermode(PorterDuff.Mode.CLEAR)
    }

    if (template.systemFolder == "nds") {
        clearDsWindowMaskPixels(bitmap)

        // DS ESRB rating inside the center label window.
        canvas.drawRoundRect(
            RectF(
                bitmap.width * 0.112f,
                bitmap.height * 0.575f,
                bitmap.width * 0.292f,
                bitmap.height * 0.825f
            ),
            bitmap.width * 0.008f,
            bitmap.width * 0.008f,
            clearPaint
        )

        // DS built-in seal.
        canvas.drawOval(
            RectF(
                bitmap.width * 0.642f,
                bitmap.height * 0.572f,
                bitmap.width * 0.840f,
                bitmap.height * 0.735f
            ),
            clearPaint
        )
    } else {
        // 3DS / 3DS w Code: clear the old seal and nearby center-window marks.
        canvas.drawRoundRect(
            RectF(
                bitmap.width * 0.600f,
                bitmap.height * 0.530f,
                bitmap.width * 0.875f,
                bitmap.height * 0.775f
            ),
            bitmap.width * 0.010f,
            bitmap.width * 0.010f,
            clearPaint
        )

        canvas.drawOval(
            RectF(
                bitmap.width * 0.620f,
                bitmap.height * 0.548f,
                bitmap.width * 0.855f,
                bitmap.height * 0.748f
            ),
            clearPaint
        )
    }

    clearPaint.xfermode = null
}

private fun clearDsWindowMaskPixels(bitmap: Bitmap) {
    val left = (bitmap.width * 0.13f).roundToInt()
    val top = (bitmap.height * 0.20f).roundToInt()
    val right = (bitmap.width * 0.88f).roundToInt()
    val bottom = (bitmap.height * 0.78f).roundToInt()

    for (y in top until bottom) {
        for (x in left until right) {
            val color = bitmap.getPixel(x, y)
            val alpha = color ushr 24
            if (alpha > 8 &&
                android.graphics.Color.red(color) < 36 &&
                android.graphics.Color.green(color) < 36 &&
                android.graphics.Color.blue(color) < 36
            ) {
                bitmap.setPixel(x, y, android.graphics.Color.TRANSPARENT)
            }
        }
    }
}

private fun drawUploadedNintendoSealOverlay(
    context: Context,
    bitmap: Bitmap,
    template: TemplateAsset,
    sealOffsetX: Float,
    sealOffsetY: Float,
    sealScale: Float
) {
    if (!template.usesNintendoSeal) return

    val seal = runCatching {
        context.assets.open("templates/Nintendo_Official_Seal.svg.png").use { BitmapFactory.decodeStream(it) }
    }.getOrNull() ?: return

    val canvas = Canvas(bitmap)
    val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG or Paint.DITHER_FLAG)

    val base = if (template.systemFolder == "nds") {
        RectF(
            bitmap.width * 0.655f,
            bitmap.height * 0.590f,
            bitmap.width * 0.830f,
            bitmap.height * 0.725f
        )
    } else {
        RectF(
            bitmap.width * 0.642f,
            bitmap.height * 0.585f,
            bitmap.width * 0.830f,
            bitmap.height * 0.732f
        )
    }

    val scaled = base.scaledFromCenter((sealScale / 100f).coerceIn(0.2f, 2.2f))
    val dx = bitmap.width * sealOffsetX / 1000f
    val dy = -bitmap.height * sealOffsetY / 1000f
    val rect = RectF(scaled.left + dx, scaled.top + dy, scaled.right + dx, scaled.bottom + dy)

    val backing = RectF(
        rect.left + rect.width() * 0.17f,
        rect.top + rect.height() * 0.22f,
        rect.right - rect.width() * 0.17f,
        rect.bottom - rect.height() * 0.22f
    )
    val backingPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = android.graphics.Color.rgb(248, 246, 236)
    }

    canvas.drawOval(backing, backingPaint)
    canvas.drawBitmap(seal, null, rect, paint)
}

private fun drawGameBoyNintendoLogoReplacement(
    context: Context,
    bitmap: Bitmap,
    template: TemplateAsset,
    showGameBoyLogo: Boolean,
    logoOffsetX: Float,
    logoOffsetY: Float,
    logoScale: Float
) {
    if (!template.supportsGameBoyLogoToggle) return

    val canvas = Canvas(bitmap)
    val clearPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        xfermode = PorterDuffXfermode(PorterDuff.Mode.CLEAR)
    }

    val clearRect = RectF(
        bitmap.width * 0.285f,
        bitmap.height * 0.690f,
        bitmap.width * 0.740f,
        bitmap.height * 0.845f
    )
    canvas.drawRoundRect(clearRect, bitmap.width * 0.025f, bitmap.width * 0.025f, clearPaint)
    clearPaint.xfermode = null

    if (!showGameBoyLogo) return

    val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG or Paint.DITHER_FLAG)
    val logo = runCatching {
        context.assets.open("templates/nintendo-logo-vector-format-11.png").use { BitmapFactory.decodeStream(it) }
    }.getOrNull()

    if (logo != null) {
        val source = logo.alphaBounds() ?: Rect(0, 0, logo.width, logo.height)
        val logoAspect = 4.05f
        val scale = (logoScale / 100f).coerceIn(0.2f, 2.2f)
        val drawWidth = bitmap.width * 0.335f * scale
        val drawHeight = drawWidth / logoAspect
        val dx = bitmap.width * logoOffsetX / 1000f
        val dy = -bitmap.height * logoOffsetY / 1000f
        val centerX = bitmap.width * 0.505f + dx
        val centerY = bitmap.height * 0.790f + dy
        val dest = RectF(
            centerX - drawWidth / 2f,
            centerY - drawHeight / 2f,
            centerX + drawWidth / 2f,
            centerY + drawHeight / 2f
        )
        canvas.drawBitmap(logo, source, dest, paint)
    } else {
        val red = android.graphics.Color.rgb(239, 35, 52)
        val scale = (logoScale / 100f).coerceIn(0.2f, 2.2f)
        val drawWidth = bitmap.width * 0.315f * scale
        val drawHeight = drawWidth / 4.05f
        val dx = bitmap.width * logoOffsetX / 1000f
        val dy = -bitmap.height * logoOffsetY / 1000f
        val centerX = bitmap.width * 0.505f + dx
        val centerY = bitmap.height * 0.790f + dy
        val dest = RectF(
            centerX - drawWidth / 2f,
            centerY - drawHeight / 2f,
            centerX + drawWidth / 2f,
            centerY + drawHeight / 2f
        )
        val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = bitmap.width * 0.010f
            color = red
        }
        val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = red
            textAlign = Paint.Align.CENTER
            typeface = Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD)
            textSize = dest.height() * 0.58f
        }
        canvas.drawRoundRect(dest, dest.height() * 0.48f, dest.height() * 0.48f, strokePaint)
        canvas.drawText("Nintendo", dest.centerX(), dest.top + dest.height() * 0.68f, textPaint)
    }
}

private data class ProductCodeDrawSpec(
    val textRegion: RectF,
    val textColor: Int,
    val fillColor: Int?,
    val textSizeRatio: Float,
    val typefaceName: String,
    val typefaceStyle: Int,
    val letterSpacingEm: Float,
    val textAlign: Paint.Align = Paint.Align.CENTER
)

private fun drawGameProductCode(
    bitmap: Bitmap,
    template: TemplateAsset,
    productCode: String,
    bold: Boolean,
    sizeStep: Int,
    offsetX: Float,
    offsetY: Float
) {
    val spec = productCodeDrawSpec(bitmap, template) ?: return
    val textRegion = RectF(spec.textRegion).apply {
        offset(bitmap.width * offsetX / 1000f, -bitmap.height * offsetY / 1000f)
    }
    val canvas = Canvas(bitmap)
    spec.fillColor?.let { fillColor ->
        val coverPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            color = fillColor
        }
        canvas.drawRoundRect(textRegion, bitmap.width * 0.004f, bitmap.width * 0.004f, coverPaint)
    }

    val textPaint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.SUBPIXEL_TEXT_FLAG).apply {
        color = spec.textColor
        textAlign = spec.textAlign
        typeface = Typeface.create(spec.typefaceName, if (bold) Typeface.BOLD else spec.typefaceStyle)
        textSize = bitmap.height * spec.textSizeRatio * (1f + sizeStep.coerceIn(-6, 8) * 0.07f)
        letterSpacing = spec.letterSpacingEm
    }
    while (textPaint.measureText(productCode) > textRegion.width() * 0.94f &&
        textPaint.textSize > bitmap.height * 0.022f
    ) {
        textPaint.textSize *= 0.92f
    }
    val bounds = Rect()
    textPaint.getTextBounds(productCode, 0, productCode.length, bounds)
    val baseline = textRegion.centerY() - bounds.exactCenterY()
    val x = when (spec.textAlign) {
        Paint.Align.LEFT -> textRegion.left
        Paint.Align.RIGHT -> textRegion.right
        else -> textRegion.centerX()
    }
    canvas.drawText(productCode, x, baseline, textPaint)
}

private fun productCodeDrawSpec(bitmap: Bitmap, template: TemplateAsset): ProductCodeDrawSpec? {
    return when {
        template.systemFolder == "n3ds" && template.supportsGameCodeGeneration ->
            ProductCodeDrawSpec(
                textRegion = RectF(
                    bitmap.width * 0.204f,
                    bitmap.height * 0.835f,
                    bitmap.width * 0.796f,
                    bitmap.height * 0.886f
                ),
                textColor = AndroidColor.rgb(18, 18, 18),
                fillColor = null,
                textSizeRatio = 0.033f,
                typefaceName = "monospace",
                typefaceStyle = Typeface.NORMAL,
                letterSpacingEm = 0.14f
            )

        template.systemFolder == "nds" ->
            ProductCodeDrawSpec(
                textRegion = RectF(
                    bitmap.width * 0.230f,
                    bitmap.height * 0.826f,
                    bitmap.width * 0.770f,
                    bitmap.height * 0.876f
                ),
                textColor = AndroidColor.rgb(18, 18, 18),
                fillColor = null,
                textSizeRatio = 0.031f,
                typefaceName = "monospace",
                typefaceStyle = Typeface.NORMAL,
                letterSpacingEm = 0.11f,
                textAlign = Paint.Align.RIGHT
            )

        template.systemFolder == "switch" ->
            ProductCodeDrawSpec(
                textRegion = RectF(
                    bitmap.width * 0.112f,
                    bitmap.height * 0.735f,
                    bitmap.width * 0.888f,
                    bitmap.height * 0.786f
                ),
                textColor = AndroidColor.WHITE,
                fillColor = null,
                textSizeRatio = 0.026f,
                typefaceName = "monospace",
                typefaceStyle = Typeface.NORMAL,
                letterSpacingEm = 0.04f
            )

        else -> null
    }
}

private fun Bitmap.alphaBounds(): Rect? {
    var left = width
    var top = height
    var right = -1
    var bottom = -1

    for (y in 0 until height) {
        for (x in 0 until width) {
            if ((getPixel(x, y) ushr 24) > 8) {
                if (x < left) left = x
                if (x > right) right = x
                if (y < top) top = y
                if (y > bottom) bottom = y
            }
        }
    }

    return if (right >= left && bottom >= top) Rect(left, top, right + 1, bottom + 1) else null
}

private fun RectF.scaledFromCenter(scale: Float): RectF {
    val halfWidth = width() * scale / 2f
    val halfHeight = height() * scale / 2f
    return RectF(centerX() - halfWidth, centerY() - halfHeight, centerX() + halfWidth, centerY() + halfHeight)
}

private fun prepareTemplateBitmap(
    context: Context,
    template: TemplateAsset,
    showGameBoyLogo: Boolean,
    show3dsSeal: Boolean,
    sealOffsetX: Float = 0f,
    sealOffsetY: Float = 0f,
    sealScale: Float = 100f,
    logoOffsetX: Float = 0f,
    logoOffsetY: Float = 0f,
    logoScale: Float = 100f,
    gameProductCode: String? = null,
    gameCodeBold: Boolean = false,
    gameCodeSizeStep: Int = 0,
    gameCodeOffsetX: Float = 0f,
    gameCodeOffsetY: Float = 0f
): Bitmap? {
    val decoded = decodeTemplateBitmap(context, template) ?: return null
    val editable = decoded.copy(Bitmap.Config.ARGB_8888, true)

    drawGameBoyNintendoLogoReplacement(context, editable, template, showGameBoyLogo, logoOffsetX, logoOffsetY, logoScale)

    if (template.supportsGameCodeGeneration && gameProductCode != null) {
        drawGameProductCode(editable, template, gameProductCode, gameCodeBold, gameCodeSizeStep, gameCodeOffsetX, gameCodeOffsetY)
    }

    if (template.usesNintendoSeal && show3dsSeal) {
        drawUploadedNintendoSealOverlay(context, editable, template, sealOffsetX, sealOffsetY, sealScale)
    }

    return editable
}








private fun hideGameBoyLogo(bitmap: Bitmap) {
    val canvas = Canvas(bitmap)
    val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        xfermode = PorterDuffXfermode(PorterDuff.Mode.CLEAR)
    }
    canvas.drawRoundRect(
        RectF(
            bitmap.width * 0.61f,
            bitmap.height * 0.795f,
            bitmap.width * 0.755f,
            bitmap.height * 0.84f
        ),
        bitmap.width * 0.02f,
        bitmap.width * 0.02f,
        paint
    )
    paint.xfermode = null
}

private fun hide3dsSeal(bitmap: Bitmap) {
    val canvas = Canvas(bitmap)
    val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        xfermode = PorterDuffXfermode(PorterDuff.Mode.CLEAR)
    }
    canvas.drawRoundRect(
        RectF(
            bitmap.width * 0.485f,
            bitmap.height * 0.44f,
            bitmap.width * 0.855f,
            bitmap.height * 0.80f
        ),
        bitmap.width * 0.02f,
        bitmap.width * 0.02f,
        paint
    )
    paint.xfermode = null
}

private fun decodeBitmap(context: Context, source: ImageSource): Bitmap? {
    return runCatching {
        val options = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        when (source) {
            is ImageSource.FileSource -> BitmapFactory.decodeFile(source.file.absolutePath, options)
            is ImageSource.UriSource -> context.contentResolver.openInputStream(source.uri)?.use {
                BitmapFactory.decodeStream(it, null, options)
            }
        }
    }.getOrNull()
}


private fun Bitmap.clearGameCubeInnerArtworkCutout() {
    val canvas = Canvas(this)
    val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        xfermode = PorterDuffXfermode(PorterDuff.Mode.CLEAR)
    }

    val cx = width / 2f
    val cy = height / 2f
    val radius = minOf(width, height) * 0.118f
    canvas.drawCircle(cx, cy, radius, paint)
    paint.xfermode = null
}

private fun Bitmap.coverDrawRect(
    targetWidth: Int,
    targetHeight: Int,
    offsetXPx: Float,
    offsetYPx: Float,
    scalePercent: Float
): RectF {
    val baseScale = max(targetWidth.toFloat() / width.toFloat(), targetHeight.toFloat() / height.toFloat())
    val scale = baseScale * (scalePercent / 100f)
    val scaledWidth = width * scale
    val scaledHeight = height * scale
    val left = (targetWidth - scaledWidth) / 2f + offsetXPx
    val top = (targetHeight - scaledHeight) / 2f - offsetYPx
    return RectF(left, top, left + scaledWidth, top + scaledHeight)
}

private fun outpaintRequestKey(
    template: TemplateAsset,
    cover: ImageAsset,
    offsetX: Float,
    offsetY: Float,
    scale: Float
): String {
    return listOf(
        template.assetPath,
        cover.source.key,
        offsetX.roundToInt().toString(),
        offsetY.roundToInt().toString(),
        scale.roundToInt().toString()
    ).joinToString("|")
}


private fun renderComposite(
    context: Context,
    templateAsset: TemplateAsset?,
    template: Bitmap?,
    artworkMask: Bitmap? = null,
    cover: Bitmap?,
    extraCover: Bitmap? = null,
    offsetXPx: Float,
    offsetYPx: Float,
    scalePercent: Float,
    extraOffsetXPx: Float = 0f,
    extraOffsetYPx: Float = 0f,
    extraScalePercent: Float = 100f,
    outpaintArtwork: Boolean,
    outpaintFill: Bitmap? = null,
    drawTemplateOverlay: Boolean = true,
    blockCenterCutout: Boolean = false
): Bitmap? {
    val width = template?.width ?: cover?.width ?: extraCover?.width ?: return null
    val height = template?.height ?: cover?.height ?: extraCover?.height ?: return null
    val output = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(output)
    val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG or Paint.DITHER_FLAG)
    canvas.drawColor(android.graphics.Color.TRANSPARENT, PorterDuff.Mode.CLEAR)

    if (cover != null || extraCover != null) {
        val coverLayer = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val coverCanvas = Canvas(coverLayer)
        val coverRect = cover?.coverDrawRect(width, height, offsetXPx, offsetYPx, scalePercent)
        val artworkWindow = templateAsset?.artworkWindow(context, width, height)
        if (outpaintArtwork && artworkWindow != null && coverRect != null && !coverRect.contains(artworkWindow)) {
            outpaintFill?.let { fill ->
                coverCanvas.drawBitmap(fill, null, artworkWindow, paint)
            }
        }
        if (cover != null && coverRect != null) {
            coverCanvas.drawBitmap(cover, null, coverRect, paint)
        }
        if (extraCover != null) {
            val extraRect = extraCover.coverDrawRect(width, height, extraOffsetXPx, extraOffsetYPx, extraScalePercent)
            coverCanvas.drawBitmap(extraCover, null, extraRect, paint)
        }

        if (template != null) {
            val maskPaint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG).apply {
                xfermode = PorterDuffXfermode(PorterDuff.Mode.DST_IN)
            }
            val mask = artworkMask ?: createArtworkMask(template, templateAsset)
            coverCanvas.drawBitmap(mask, null, RectF(0f, 0f, width.toFloat(), height.toFloat()), maskPaint)
            maskPaint.xfermode = null
        }

        if ((blockCenterCutout || templateAsset?.isGameCubeTemplate == true) && templateAsset?.isDiscTemplate != true) {
            coverLayer.clearGameCubeInnerArtworkCutout()
        }
        canvas.drawBitmap(coverLayer, 0f, 0f, paint)
    }

    if (template != null && drawTemplateOverlay) {
        canvas.drawBitmap(template, null, RectF(0f, 0f, width.toFloat(), height.toFloat()), paint)
    }

    return output
}

private fun createCartridgeMask(template: Bitmap): Bitmap {
    val width = template.width
    val height = template.height
    val totalPixels = width * height
    val sourcePixels = IntArray(totalPixels)
    template.getPixels(sourcePixels, 0, width, 0, 0, width, height)

    val outside = BooleanArray(totalPixels)
    val queue = java.util.ArrayDeque<Int>()

    fun enqueueIfOutside(index: Int) {
        if (!outside[index] && (sourcePixels[index] ushr 24) <= 8) {
            outside[index] = true
            queue.add(index)
        }
    }

    for (x in 0 until width) {
        enqueueIfOutside(x)
        enqueueIfOutside((height - 1) * width + x)
    }
    for (y in 0 until height) {
        enqueueIfOutside(y * width)
        enqueueIfOutside(y * width + width - 1)
    }

    while (!queue.isEmpty()) {
        val index = queue.removeFirst()
        val x = index % width
        val y = index / width
        if (x > 0) enqueueIfOutside(index - 1)
        if (x < width - 1) enqueueIfOutside(index + 1)
        if (y > 0) enqueueIfOutside(index - width)
        if (y < height - 1) enqueueIfOutside(index + width)
    }

    var interior = BooleanArray(totalPixels) { !outside[it] }
    repeat(MASK_INSET_PIXELS) {
        val inset = interior.copyOf()
        for (y in 0 until height) {
            for (x in 0 until width) {
                val index = y * width + x
                if (!interior[index]) continue
                val touchesEdge = x == 0 || x == width - 1 || y == 0 || y == height - 1
                val touchesOutside = touchesEdge ||
                    !interior[index - 1] ||
                    !interior[index + 1] ||
                    !interior[index - width] ||
                    !interior[index + width]
                if (touchesOutside) {
                    inset[index] = false
                }
            }
        }
        interior = inset
    }

    val maskPixels = IntArray(totalPixels)
    for (index in 0 until totalPixels) {
        if (interior[index]) {
            maskPixels[index] = android.graphics.Color.WHITE
        }
    }

    return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).apply {
        setPixels(maskPixels, 0, width, 0, 0, width, height)
    }
}

private fun createArtworkMask(template: Bitmap, templateAsset: TemplateAsset?): Bitmap {
    if (templateAsset?.isDiscTemplate == true) {
        return createDiscArtworkMask(template.width, template.height, templateAsset)
    }
    return createCartridgeMask(template)
}

private fun createDiscArtworkMask(width: Int, height: Int, template: TemplateAsset): Bitmap {
    val output = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(output)
    val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = android.graphics.Color.WHITE
        style = Paint.Style.FILL
    }
    val cx = width / 2f
    val cy = height / 2f
    val outerRadius = minOf(width, height) * 0.485f
    canvas.drawCircle(cx, cy, outerRadius, paint)

    paint.xfermode = PorterDuffXfermode(PorterDuff.Mode.CLEAR)
    canvas.drawCircle(cx, cy, template.centerHoleRadius(width, height), paint)
    paint.xfermode = null
    return output
}

private fun TemplateAsset.centerHoleRadius(width: Int, height: Int): Float {
    val base = minOf(width, height)
    return when (systemFolder) {
        "gc" -> base * 0.118f
        "ps2" -> base * 0.104f
        "ps3" -> base * 0.086f
        "psx" -> base * 0.102f
        "psp" -> base * 0.070f
        else -> base * 0.095f
    }
}

private data class FeatherMask(
    val alpha: FloatArray,
    val hardMask: BooleanArray,
    val edgeColors: IntArray,
    val ambientColor: Int,
    val width: Int,
    val height: Int
)

private fun lamaModelFile(context: Context): File {
    val dir = File(context.filesDir, "models")
    if (!dir.exists()) dir.mkdirs()
    return File(dir, LAMA_MODEL_NAME)
}

private suspend fun ensureLamaModel(
    context: Context,
    onProgress: suspend (OutpaintProgress) -> Unit
): File {
    val model = lamaModelFile(context)
    if (model.exists() && model.length() > 100_000_000L) {
        onProgress(OutpaintProgress("AI model ready", 0.52f))
        return model
    }

    val tmp = File(model.parentFile, "$LAMA_MODEL_NAME.tmp")
    tmp.delete()
    onProgress(OutpaintProgress("Downloading AI model", 0.02f))
    val connection = (URL(LAMA_MODEL_URL).openConnection() as HttpURLConnection).apply {
        connectTimeout = 20_000
        readTimeout = 60_000
        instanceFollowRedirects = true
        setRequestProperty("User-Agent", "CartridgeStamper/1.0")
    }

    connection.inputStream.use { input ->
        tmp.outputStream().use { output ->
            val buffer = ByteArray(1024 * 1024)
            val totalBytes = connection.contentLengthLong.takeIf { it > 0L }
            var downloadedBytes = 0L
            while (true) {
                val read = input.read(buffer)
                if (read <= 0) break
                output.write(buffer, 0, read)
                downloadedBytes += read
                val fraction = totalBytes?.let { total ->
                    0.02f + 0.50f * (downloadedBytes.toFloat() / total.toFloat()).coerceIn(0f, 1f)
                } ?: 0.18f
                onProgress(OutpaintProgress("Downloading AI model", fraction))
            }
        }
    }

    if (tmp.length() <= 100_000_000L) {
        tmp.delete()
        error("LaMa model download was incomplete")
    }

    if (model.exists()) model.delete()
    if (!tmp.renameTo(model)) {
        tmp.copyTo(model, overwrite = true)
        tmp.delete()
    }
    onProgress(OutpaintProgress("AI model ready", 0.54f))
    return model
}

private fun getOrCreateLamaSession(model: File): Pair<OrtEnvironment, OrtSession> {
    synchronized(lamaRuntimeLock) {
        val env = lamaOrtEnv ?: OrtEnvironment.getEnvironment().also { lamaOrtEnv = it }
        val existing = lamaOrtSession
        if (existing != null && lamaOrtModelPath == model.absolutePath) {
            return env to existing
        }

        runCatching { lamaOrtSession?.close() }
        lamaOrtSession = null
        lamaOrtModelPath = null

        val session = runCatching {
            createLamaSession(env, model, preferXnnpack = true)
        }.onFailure {
            Log.w("CartridgeStamperAI", "XNNPACK LaMa session unavailable; falling back to CPU", it)
        }.getOrElse {
            createLamaSession(env, model, preferXnnpack = false)
        }

        lamaOrtSession = session
        lamaOrtModelPath = model.absolutePath
        return env to session
    }
}

private fun createLamaSession(
    env: OrtEnvironment,
    model: File,
    preferXnnpack: Boolean
): OrtSession {
    val cpuThreads = (Runtime.getRuntime().availableProcessors() - 1).coerceIn(2, 4)
    val options = OrtSession.SessionOptions()
    try {
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        options.setMemoryPatternOptimization(true)
        options.setCPUArenaAllocator(true)
        options.setIntraOpNumThreads(cpuThreads)
        options.setInterOpNumThreads(1)
        if (preferXnnpack) {
            options.addXnnpack(mapOf("intra_op_num_threads" to cpuThreads.toString()))
        }
        return env.createSession(model.absolutePath, options)
    } finally {
        options.close()
    }
}

private suspend fun generateLamaOutpaintForArtworkWindow(
    context: Context,
    cover: Bitmap,
    artworkWindow: RectF,
    coverRect: RectF,
    onProgress: suspend (OutpaintProgress) -> Unit
): Bitmap {
    onProgress(OutpaintProgress("Preparing outpaint", 0.01f))
    val windowWidth = artworkWindow.width().roundToInt().coerceAtLeast(1)
    val windowHeight = artworkWindow.height().roundToInt().coerceAtLeast(1)
    val sourceWindow = Bitmap.createBitmap(windowWidth, windowHeight, Bitmap.Config.ARGB_8888)
    val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG or Paint.DITHER_FLAG)
    Canvas(sourceWindow).apply {
        drawColor(AndroidColor.TRANSPARENT, PorterDuff.Mode.CLEAR)
        drawBitmap(
            cover,
            null,
            RectF(
                coverRect.left - artworkWindow.left,
                coverRect.top - artworkWindow.top,
                coverRect.right - artworkWindow.left,
                coverRect.bottom - artworkWindow.top
            ),
            paint
        )
    }

    val fillMask = createOutpaintMaskFromEmptyEdges(sourceWindow)
    if (!fillMask.hasMaskedPixels()) {
        fillMask.recycle()
        onProgress(OutpaintProgress("Outpaint not needed", 1f))
        return sourceWindow
    }

    val model = ensureLamaModel(context, onProgress)
    onProgress(OutpaintProgress("Preparing AI input", 0.60f))
    val aiW = 512
    val aiH = 512
    var imageForAi: Bitmap? = null
    var maskForAi: Bitmap? = null
    var generatedSmall: Bitmap? = null
    var generatedFull: Bitmap? = null

    return try {
        val preparedImage = Bitmap.createScaledBitmap(sourceWindow, aiW, aiH, true).also {
            imageForAi = it
        }
        val aiMaskBase = Bitmap.createScaledBitmap(fillMask, aiW, aiH, false)
        val preparedMask = try {
            expandMaskForGeneration(aiMaskBase, 0)
        } finally {
            aiMaskBase.recycle()
        }.also {
            maskForAi = it
        }

        val imageTensorData = FloatArray(1 * 3 * aiH * aiW)
        val maskTensorData = FloatArray(1 * 1 * aiH * aiW)
        val imagePixels = IntArray(aiW * aiH)
        val maskPixels = IntArray(aiW * aiH)
        preparedImage.getPixels(imagePixels, 0, aiW, 0, 0, aiW, aiH)
        preparedMask.getPixels(maskPixels, 0, aiW, 0, 0, aiW, aiH)

        val hardAiMask = BooleanArray(aiW * aiH)
        for (i in maskPixels.indices) {
            val color = maskPixels[i]
            val white = (AndroidColor.red(color) + AndroidColor.green(color) + AndroidColor.blue(color)) / 3
            hardAiMask[i] = white > 127
        }

        val inputFallback = averageUnmaskedColor(hardAiMask, imagePixels)
        val inputEdgeColors = createEdgeColorGuide(
            hardMask = hardAiMask,
            originalPixels = imagePixels,
            width = aiW,
            height = aiH,
            radius = aiW + aiH,
            maskedFallback = inputFallback
        )

        for (y in 0 until aiH) {
            for (x in 0 until aiW) {
                val idx = y * aiW + x
                val color = if (hardAiMask[idx]) inputEdgeColors[idx] else imagePixels[idx]
                maskTensorData[idx] = if (hardAiMask[idx]) 1f else 0f
                imageTensorData[idx] = AndroidColor.red(color) / 255f
                imageTensorData[aiH * aiW + idx] = AndroidColor.green(color) / 255f
                imageTensorData[2 * aiH * aiW + idx] = AndroidColor.blue(color) / 255f
            }
        }

        onProgress(OutpaintProgress("Loading AI model", 0.68f))
        val (env, session) = getOrCreateLamaSession(model)
        onProgress(OutpaintProgress("Processing outpaint", 0.76f))
        OnnxTensor.createTensor(env, FloatBuffer.wrap(imageTensorData), longArrayOf(1, 3, aiH.toLong(), aiW.toLong())).use { imageTensor ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(maskTensorData), longArrayOf(1, 1, aiH.toLong(), aiW.toLong())).use { maskTensor ->
                session.run(
                    mapOf(
                        "image" to imageTensor,
                        "mask" to maskTensor
                    )
                ).use { results ->
                    val outputTensor = results[0] as OnnxTensor
                    val outputBuffer = outputTensor.floatBuffer
                    val channelSize = aiW * aiH
                    var outMax = Float.NEGATIVE_INFINITY
                    for (i in 0 until channelSize * 3) {
                        val value = outputBuffer.get(i)
                        if (value > outMax) outMax = value
                    }

                    val outputScale = if (outMax <= 2.0f) 255f else 1f
                    val outPixels = IntArray(channelSize)
                    for (i in 0 until channelSize) {
                        val r = (outputBuffer.get(i) * outputScale).toInt().coerceIn(0, 255)
                        val g = (outputBuffer.get(channelSize + i) * outputScale).toInt().coerceIn(0, 255)
                        val b = (outputBuffer.get(channelSize * 2 + i) * outputScale).toInt().coerceIn(0, 255)
                        outPixels[i] = AndroidColor.argb(255, r, g, b)
                    }

                    val smallGenerated = Bitmap.createBitmap(aiW, aiH, Bitmap.Config.ARGB_8888).apply {
                        setPixels(outPixels, 0, aiW, 0, 0, aiW, aiH)
                    }
                    onProgress(OutpaintProgress("Blending outpaint", 0.92f))
                    generatedSmall = smallGenerated
                    val fullGenerated = Bitmap.createScaledBitmap(smallGenerated, sourceWindow.width, sourceWindow.height, true)
                    generatedFull = fullGenerated
                    val featherMask = createScaledFeatherMask(fillMask, sourceWindow, aiW, aiH, 18)
                    blendGeneratedIntoMask(sourceWindow, fullGenerated, featherMask)
                }
            }
        }
    } finally {
        fillMask.recycle()
        imageForAi?.recycle()
        maskForAi?.recycle()
        generatedSmall?.recycle()
        generatedFull?.recycle()
        sourceWindow.recycle()
    }
}

private fun Bitmap.hasMaskedPixels(): Boolean {
    val pixels = IntArray(width * height)
    getPixels(pixels, 0, width, 0, 0, width, height)
    return pixels.any {
        (AndroidColor.red(it) + AndroidColor.green(it) + AndroidColor.blue(it)) / 3 > 127
    }
}

private fun createOutpaintMaskFromEmptyEdges(source: Bitmap): Bitmap {
    val width = source.width
    val height = source.height
    val pixels = IntArray(width * height)
    source.getPixels(pixels, 0, width, 0, 0, width, height)

    var contentLeft = width
    var contentTop = height
    var contentRight = -1
    var contentBottom = -1

    for (y in 0 until height) {
        val row = y * width
        for (x in 0 until width) {
            val color = pixels[row + x]
            if (AndroidColor.alpha(color) >= 16) {
                if (x < contentLeft) contentLeft = x
                if (x > contentRight) contentRight = x
                if (y < contentTop) contentTop = y
                if (y > contentBottom) contentBottom = y
            }
        }
    }

    val mask = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    val maskPixels = IntArray(width * height) { AndroidColor.BLACK }
    if (contentRight < contentLeft || contentBottom < contentTop) {
        mask.setPixels(maskPixels, 0, width, 0, 0, width, height)
        return mask
    }

    val seamOverlap = 3
    val maskLeft = (contentLeft + seamOverlap).coerceIn(0, width - 1)
    val maskTop = (contentTop + seamOverlap).coerceIn(0, height - 1)
    val maskRight = (contentRight - seamOverlap).coerceIn(0, width - 1)
    val maskBottom = (contentBottom - seamOverlap).coerceIn(0, height - 1)

    for (y in 0 until height) {
        val row = y * width
        for (x in 0 until width) {
            val outsideArtwork = x < maskLeft || x > maskRight || y < maskTop || y > maskBottom
            if (outsideArtwork) maskPixels[row + x] = AndroidColor.WHITE
        }
    }

    mask.setPixels(maskPixels, 0, width, 0, 0, width, height)
    return mask
}

private fun expandMaskForGeneration(mask: Bitmap, radius: Int): Bitmap {
    val width = mask.width
    val height = mask.height
    val src = IntArray(width * height)
    val dst = IntArray(width * height)
    mask.getPixels(src, 0, width, 0, 0, width, height)

    for (y in 0 until height) {
        val yMin = (y - radius).coerceAtLeast(0)
        val yMax = (y + radius).coerceAtMost(height - 1)
        for (x in 0 until width) {
            val xMin = (x - radius).coerceAtLeast(0)
            val xMax = (x + radius).coerceAtMost(width - 1)
            var shouldMask = false
            loop@ for (yy in yMin..yMax) {
                val row = yy * width
                for (xx in xMin..xMax) {
                    val c = src[row + xx]
                    val white = (AndroidColor.red(c) + AndroidColor.green(c) + AndroidColor.blue(c)) / 3
                    if (white > 127) {
                        shouldMask = true
                        break@loop
                    }
                }
            }
            dst[y * width + x] = if (shouldMask) AndroidColor.WHITE else AndroidColor.BLACK
        }
    }

    return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).apply {
        setPixels(dst, 0, width, 0, 0, width, height)
    }
}

private fun createScaledFeatherMask(
    mask: Bitmap,
    original: Bitmap,
    width: Int,
    height: Int,
    radius: Int
): FeatherMask {
    val scaledMask = Bitmap.createScaledBitmap(mask, width, height, false)
    val scaledOriginal = Bitmap.createScaledBitmap(original, width, height, true)
    return try {
        val maskPixels = IntArray(width * height)
        val originalPixels = IntArray(width * height)
        scaledMask.getPixels(maskPixels, 0, width, 0, 0, width, height)
        scaledOriginal.getPixels(originalPixels, 0, width, 0, 0, width, height)

        val hardMask = BooleanArray(width * height)
        for (i in maskPixels.indices) {
            val color = maskPixels[i]
            val white = (AndroidColor.red(color) + AndroidColor.green(color) + AndroidColor.blue(color)) / 3
            hardMask[i] = white > 127
        }
        val ambientColor = averageUnmaskedColor(hardMask, originalPixels)

        FeatherMask(
            alpha = createFeatheredCompositeMask(scaledMask, radius),
            hardMask = hardMask,
            edgeColors = createEdgeColorGuide(
                hardMask = hardMask,
                originalPixels = originalPixels,
                width = width,
                height = height,
                radius = width + height,
                maskedFallback = ambientColor
            ),
            ambientColor = ambientColor,
            width = width,
            height = height
        )
    } finally {
        scaledMask.recycle()
        scaledOriginal.recycle()
    }
}

private fun createEdgeColorGuide(
    hardMask: BooleanArray,
    originalPixels: IntArray,
    width: Int,
    height: Int,
    radius: Int,
    maskedFallback: Int? = null
): IntArray {
    val edgeColors = if (maskedFallback != null) {
        IntArray(originalPixels.size) { index ->
            if (hardMask[index]) maskedFallback else originalPixels[index]
        }
    } else {
        originalPixels.copyOf()
    }

    val maxSteps = radius.coerceAtLeast(0)
    if (maxSteps == 0) return edgeColors

    val size = width * height
    val distance = IntArray(size) { -1 }
    val queue = IntArray(size)
    var head = 0
    var tail = 0

    for (i in 0 until size) {
        if (!hardMask[i]) {
            distance[i] = 0
            queue[tail++] = i
        }
    }

    fun visit(from: Int, to: Int) {
        if (!hardMask[to] || distance[to] >= 0) return
        val nextDistance = distance[from] + 1
        if (nextDistance > maxSteps) return
        distance[to] = nextDistance
        edgeColors[to] = edgeColors[from]
        queue[tail++] = to
    }

    while (head < tail) {
        val index = queue[head++]
        val step = distance[index]
        if (step >= maxSteps) continue
        val x = index % width
        val y = index / width
        if (x > 0) visit(index, index - 1)
        if (x < width - 1) visit(index, index + 1)
        if (y > 0) visit(index, index - width)
        if (y < height - 1) visit(index, index + width)
    }

    return edgeColors
}

private fun averageUnmaskedColor(hardMask: BooleanArray, pixels: IntArray): Int {
    var rTotal = 0L
    var gTotal = 0L
    var bTotal = 0L
    var count = 0L

    for (i in pixels.indices) {
        if (hardMask[i]) continue
        val color = pixels[i]
        if (AndroidColor.alpha(color) < 16) continue
        rTotal += AndroidColor.red(color)
        gTotal += AndroidColor.green(color)
        bTotal += AndroidColor.blue(color)
        count++
    }

    if (count == 0L) return AndroidColor.rgb(32, 32, 32)
    return AndroidColor.rgb(
        (rTotal / count).toInt().coerceIn(0, 255),
        (gTotal / count).toInt().coerceIn(0, 255),
        (bTotal / count).toInt().coerceIn(0, 255)
    )
}

private fun blendGeneratedIntoMask(
    original: Bitmap,
    generated: Bitmap,
    featherMask: FeatherMask
): Bitmap {
    val width = original.width
    val height = original.height
    val resultPixels = IntArray(width * height)
    val generatedPixels = IntArray(width * height)
    original.getPixels(resultPixels, 0, width, 0, 0, width, height)
    generated.getPixels(generatedPixels, 0, width, 0, 0, width, height)

    for (y in 0 until height) {
        val resultRow = y * width
        val maskY = (y * featherMask.height / height).coerceIn(0, featherMask.height - 1)
        val maskRow = maskY * featherMask.width
        for (x in 0 until width) {
            val maskX = (x * featherMask.width / width).coerceIn(0, featherMask.width - 1)
            val maskIndex = maskRow + maskX
            if (featherMask.hardMask[maskIndex]) {
                val index = resultRow + x
                val seam = featherMask.alpha[maskIndex].coerceIn(0f, 1f)
                resultPixels[index] = suppressGeneratedShadow(
                    generated = generatedPixels[index],
                    edge = featherMask.edgeColors[maskIndex],
                    seam = seam
                )
            }
        }
    }

    return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).apply {
        setPixels(resultPixels, 0, width, 0, 0, width, height)
    }
}

private fun suppressGeneratedShadow(generated: Int, edge: Int, seam: Float): Int {
    val generatedLuma = luma(generated)
    val edgeLuma = luma(edge)
    val seamWeight = (1f - seam).coerceIn(0f, 1f)
    val hardBlackSeam = seamWeight > 0.20f && generatedLuma < 8f && edgeLuma > 18f
    if (!hardBlackSeam) return generated
    return blendColors(generated, edge, (seamWeight * 0.28f).coerceIn(0f, 0.32f))
}

private fun createFeatheredCompositeMask(mask: Bitmap, radius: Int): FloatArray {
    val width = mask.width
    val height = mask.height
    val src = IntArray(width * height)
    mask.getPixels(src, 0, width, 0, 0, width, height)
    val hard = FloatArray(width * height)
    for (i in src.indices) {
        val c = src[i]
        val white = (AndroidColor.red(c) + AndroidColor.green(c) + AndroidColor.blue(c)) / 3
        hard[i] = if (white > 127) 1f else 0f
    }

    val feathered = FloatArray(width * height)
    for (y in 0 until height) {
        val yMin = (y - radius).coerceAtLeast(0)
        val yMax = (y + radius).coerceAtMost(height - 1)
        for (x in 0 until width) {
            val idx = y * width + x
            if (hard[idx] <= 0f) {
                feathered[idx] = 0f
                continue
            }

            var nearestKeep = radius + 1
            loop@ for (yy in yMin..yMax) {
                val row = yy * width
                for (xx in (x - radius).coerceAtLeast(0)..(x + radius).coerceAtMost(width - 1)) {
                    if (hard[row + xx] <= 0f) {
                        val dx = xx - x
                        val dy = yy - y
                        val dist = kotlin.math.sqrt((dx * dx + dy * dy).toFloat()).toInt()
                        if (dist < nearestKeep) nearestKeep = dist
                        if (nearestKeep == 0) break@loop
                    }
                }
            }

            feathered[idx] = when {
                nearestKeep >= radius -> 1f
                nearestKeep <= 0 -> 0f
                else -> (nearestKeep.toFloat() / radius.toFloat()).coerceIn(0f, 1f)
            }
        }
    }
    return feathered
}

private fun blendColors(original: Int, generated: Int, alpha: Float): Int {
    val a = alpha.coerceIn(0f, 1f)
    val inv = 1f - a
    val r = (AndroidColor.red(original) * inv + AndroidColor.red(generated) * a).toInt().coerceIn(0, 255)
    val g = (AndroidColor.green(original) * inv + AndroidColor.green(generated) * a).toInt().coerceIn(0, 255)
    val b = (AndroidColor.blue(original) * inv + AndroidColor.blue(generated) * a).toInt().coerceIn(0, 255)
    return AndroidColor.argb(255, r, g, b)
}

private fun luma(color: Int): Float {
    return AndroidColor.red(color) * 0.2126f + AndroidColor.green(color) * 0.7152f + AndroidColor.blue(color) * 0.0722f
}

private fun createSoftOutpaintBackground(source: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
    val smallWidth = max(24, targetWidth / 18)
    val smallHeight = max(24, targetHeight / 18)
    val tiny = Bitmap.createBitmap(smallWidth, smallHeight, Bitmap.Config.ARGB_8888)
    Canvas(tiny).drawBitmap(
        source,
        null,
        source.coverRectFor(smallWidth, smallHeight),
        Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG or Paint.DITHER_FLAG)
    )

    val output = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
    Canvas(output).drawBitmap(
        tiny,
        null,
        RectF(0f, 0f, targetWidth.toFloat(), targetHeight.toFloat()),
        Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG or Paint.DITHER_FLAG)
    )
    return output
}

private fun Bitmap.coverRectFor(targetWidth: Int, targetHeight: Int): RectF {
    val scale = max(targetWidth.toFloat() / width.toFloat(), targetHeight.toFloat() / height.toFloat())
    val drawWidth = width * scale
    val drawHeight = height * scale
    val left = (targetWidth - drawWidth) / 2f
    val top = (targetHeight - drawHeight) / 2f
    return RectF(left, top, left + drawWidth, top + drawHeight)
}

private fun exportToDefaultFolder(
    context: Context,
    bitmap: Bitmap,
    template: TemplateAsset,
    cover: ImageAsset?
): Result<File> = runCatching {
    if (needsAllFilesAccess()) {
        error("Files access is required for exports")
    }
    val outputDirectory = template.defaultOutputDirectory()
    outputDirectory.mkdirs()
    val outputFile = File(outputDirectory, makeExportName(template, cover))
    FileOutputStream(outputFile).use {
        val cocoonReadyBitmap = bitmap.toCocoonSquareIconCanvas()
        cocoonReadyBitmap.compress(Bitmap.CompressFormat.PNG, 100, it)
    }
    MediaScannerConnection.scanFile(context, arrayOf(outputFile.absolutePath), arrayOf("image/png"), null)
    outputFile
}

private fun writeBitmapToUri(context: Context, bitmap: Bitmap, uri: Uri): Result<Unit> = runCatching {
    context.contentResolver.openOutputStream(uri)?.use {
        val cocoonReadyBitmap = bitmap.toCocoonSquareIconCanvas()
        cocoonReadyBitmap.compress(Bitmap.CompressFormat.PNG, 100, it)
    } ?: error("Could not open export destination")
}

private fun makeExportName(template: TemplateAsset?, cover: ImageAsset?): String {
    val imagePart = cover?.label?.substringBeforeLast('.')?.safeOutputNamePart()
        ?: template?.label?.safeOutputNamePart()
        ?: "cartridge"
    return "$imagePart.png"
}

private fun String.safeOutputNamePart(): String {
    return replace(Regex("""[\\/:*?"<>|]+"""), "-")
        .trim()
        .ifBlank { "image" }
}

private fun String.sanitizeFilePart(): String {
    return lowercase(Locale.US)
        .replace(Regex("[^a-z0-9]+"), "-")
        .trim('-')
        .ifBlank { "image" }
}

private fun needsAllFilesAccess(): Boolean {
    return Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && !Environment.isExternalStorageManager()
}

private fun openAllFilesSettings(context: Context) {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
        openManageAllFilesSettings(context)
    }
}

@RequiresApi(Build.VERSION_CODES.R)
private fun openManageAllFilesSettings(context: Context) {
    val packageUri = Uri.parse("package:${context.packageName}")
    val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, packageUri)
    runCatching {
        context.startActivity(intent)
    }.recoverCatching {
        context.startActivity(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION))
    }
}

private fun persistReadPermission(context: Context, uri: Uri) {
    runCatching {
        context.contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
}

private fun persistTreePermission(context: Context, uri: Uri) {
    runCatching {
        context.contentResolver.takePersistableUriPermission(
            uri,
            Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION
        )
    }
}

private fun queryDisplayName(context: Context, uri: Uri): String? {
    return context.contentResolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
        if (cursor.moveToFirst()) {
            cursor.getString(0)
        } else {
            null
        }
    }
}
