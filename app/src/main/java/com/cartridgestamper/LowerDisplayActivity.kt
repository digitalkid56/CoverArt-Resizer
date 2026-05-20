package com.cartridgestamper

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Build
import android.os.Bundle
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.Window
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.ImageView
import androidx.activity.ComponentActivity
import androidx.core.view.WindowCompat
import kotlin.math.min

class LowerDisplayActivity : ComponentActivity() {
    private lateinit var preview: ImageView
    private lateinit var scaleDetector: ScaleGestureDetector
    private lateinit var gestureDetector: GestureDetector
    private var currentBitmap: Bitmap? = null
    private var lastTouchX = 0f
    private var lastTouchY = 0f
    private var dragging = false

    private val previewReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val command = intent?.getStringExtra(EXTRA_PREVIEW_COMMAND) ?: return
            if (command != COMMAND_UPDATE_LOWER_PREVIEW) return
            showPreviewFromIntent(intent)
        }
    }

    private val closeReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        requestWindowFeature(Window.FEATURE_NO_TITLE)
        window.setFlags(
            WindowManager.LayoutParams.FLAG_FULLSCREEN,
            WindowManager.LayoutParams.FLAG_FULLSCREEN
        )
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        hideSystemBars()

        scaleDetector = ScaleGestureDetector(
            this,
            object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
                override fun onScale(detector: ScaleGestureDetector): Boolean {
                    sendBottomLayerGesture(scaleFactor = detector.scaleFactor)
                    return true
                }
            }
        )
        gestureDetector = GestureDetector(
            this,
            object : GestureDetector.SimpleOnGestureListener() {
                override fun onDown(e: MotionEvent): Boolean = true

                override fun onDoubleTap(e: MotionEvent): Boolean {
                    returnToSingleDisplay()
                    return true
                }

                override fun onSingleTapConfirmed(e: MotionEvent): Boolean = true
            }
        )

        preview = ImageView(this).apply {
            setBackgroundColor(Color.TRANSPARENT)
            scaleType = ImageView.ScaleType.FIT_CENTER
            adjustViewBounds = false
            setOnTouchListener { _, event ->
                gestureDetector.onTouchEvent(event)
                scaleDetector.onTouchEvent(event)
                when (event.actionMasked) {
                    MotionEvent.ACTION_DOWN -> {
                        dragging = true
                        lastTouchX = event.x
                        lastTouchY = event.y
                    }

                    MotionEvent.ACTION_MOVE -> {
                        if (dragging && event.pointerCount == 1 && !scaleDetector.isInProgress) {
                            val fitScale = previewFitScale()
                            if (fitScale > 0f) {
                                sendBottomLayerGesture(
                                    dx = (event.x - lastTouchX) / fitScale,
                                    dy = -(event.y - lastTouchY) / fitScale
                                )
                            }
                            lastTouchX = event.x
                            lastTouchY = event.y
                        }
                    }

                    MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                        dragging = false
                    }
                }
                true
            }
        }

        setContentView(
            FrameLayout(this).apply {
                setBackgroundColor(Color.rgb(42, 45, 49))
                addView(
                    WorkspaceGridView(this@LowerDisplayActivity),
                    FrameLayout.LayoutParams(
                        FrameLayout.LayoutParams.MATCH_PARENT,
                        FrameLayout.LayoutParams.MATCH_PARENT
                    )
                )
                addView(
                    preview,
                    FrameLayout.LayoutParams(
                        FrameLayout.LayoutParams.MATCH_PARENT,
                        FrameLayout.LayoutParams.MATCH_PARENT
                    )
                )
            }
        )

        registerReceivers()
        showPreviewFromIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        showPreviewFromIntent(intent)
    }

    override fun onDestroy() {
        runCatching { unregisterReceiver(previewReceiver) }
        runCatching { unregisterReceiver(closeReceiver) }
        super.onDestroy()
    }

    private fun registerReceivers() {
        val previewFilter = IntentFilter(ACTION_LOWER_PREVIEW)
        val closeFilter = IntentFilter(ACTION_CLOSE_LOWER_DISPLAY)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(previewReceiver, previewFilter, RECEIVER_NOT_EXPORTED)
            registerReceiver(closeReceiver, closeFilter, RECEIVER_NOT_EXPORTED)
        } else {
            @Suppress("DEPRECATION")
            registerReceiver(previewReceiver, previewFilter)
            @Suppress("DEPRECATION")
            registerReceiver(closeReceiver, closeFilter)
        }
    }

    private fun showPreviewFromIntent(intent: Intent?) {
        intent ?: return
        val bitmap = intent.getStringExtra(EXTRA_BITMAP_FILE)
            ?.let { BitmapFactory.decodeFile(it) }
            ?: intent.getByteArrayExtra(EXTRA_BITMAP_BYTES)
                ?.let { BitmapFactory.decodeByteArray(it, 0, it.size) }
            ?: return
        val shouldReset = currentBitmap?.let { it.width != bitmap.width || it.height != bitmap.height } ?: true
        currentBitmap = bitmap
        preview.setImageBitmap(bitmap)
        if (shouldReset) {
            lastTouchX = 0f
            lastTouchY = 0f
        }
    }

    private fun previewFitScale(): Float {
        val bitmap = currentBitmap ?: return 0f
        val viewWidth = preview.width
        val viewHeight = preview.height
        if (viewWidth <= 0 || viewHeight <= 0 || bitmap.width <= 0 || bitmap.height <= 0) return 0f
        return min(
            viewWidth.toFloat() / bitmap.width.toFloat(),
            viewHeight.toFloat() / bitmap.height.toFloat()
        )
    }

    private fun sendBottomLayerGesture(
        dx: Float = 0f,
        dy: Float = 0f,
        scaleFactor: Float = 1f
    ) {
        sendBroadcast(
            Intent(ACTION_LOWER_PREVIEW_GESTURE).apply {
                setPackage(packageName)
                putExtra(EXTRA_GESTURE_DX, dx)
                putExtra(EXTRA_GESTURE_DY, dy)
                putExtra(EXTRA_GESTURE_SCALE, scaleFactor)
            }
        )
    }

    private fun returnToSingleDisplay() {
        sendBroadcast(
            Intent(ACTION_LOWER_PREVIEW_TAPPED).apply {
                setPackage(packageName)
            }
        )
        finish()
    }

    private fun hideSystemBars() {
        window.decorView.systemUiVisibility =
            android.view.View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or
                android.view.View.SYSTEM_UI_FLAG_FULLSCREEN or
                android.view.View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                android.view.View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN or
                android.view.View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION or
                android.view.View.SYSTEM_UI_FLAG_LAYOUT_STABLE
    }

    private class WorkspaceGridView(context: Context) : android.view.View(context) {
        private val backgroundPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.rgb(42, 45, 49)
            style = Paint.Style.FILL
        }
        private val minorPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.rgb(58, 63, 69)
            strokeWidth = 1f
        }
        private val majorPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.rgb(85, 91, 98)
            strokeWidth = 1.5f
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), backgroundPaint)
            val minor = 24f * resources.displayMetrics.density
            var x = 0f
            var xIndex = 0
            while (x <= width) {
                canvas.drawLine(x, 0f, x, height.toFloat(), if (xIndex % 4 == 0) majorPaint else minorPaint)
                x += minor
                xIndex++
            }
            var y = 0f
            var yIndex = 0
            while (y <= height) {
                canvas.drawLine(0f, y, width.toFloat(), y, if (yIndex % 4 == 0) majorPaint else minorPaint)
                y += minor
                yIndex++
            }
        }
    }
}
