from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
import uuid
from datetime import datetime, timezone
import requests
from web3 import Web3
import qrcode
from io import BytesIO
import base64
from eth_account import Account

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Polygon Mumbai RPC
MUMBAI_RPC = "https://rpc-mumbai.maticvigil.com"
w3 = Web3(Web3.HTTPProvider(MUMBAI_RPC))

# Standard USDC contract on Polygon Mumbai
USDC_CONTRACT_ADDRESS = "0x0FA8781a83E46826621b3BC094Ea2A0212e71B23"

# Generate a placeholder wallet for receiving payments
# In production, replace this with your actual wallet private key
account = Account.create()
PAYMENT_WALLET_ADDRESS = account.address
PAYMENT_WALLET_PRIVATE_KEY = account.key.hex()

print(f"\n=== PAYMENT WALLET INFO ===")
print(f"Address: {PAYMENT_WALLET_ADDRESS}")
print(f"Private Key: {PAYMENT_WALLET_PRIVATE_KEY}")
print(f"NOTE: This is a placeholder wallet. Replace with your actual wallet in production.")
print(f"========================\n")

# USDC ABI (minimal - only what we need)
USDC_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function"
    }
]

usdc_contract = w3.eth.contract(address=Web3.to_checksum_address(USDC_CONTRACT_ADDRESS), abi=USDC_ABI)

# Models
class Order(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    eur_amount: float
    usdc_amount: float
    payment_address: str
    status: str = "awaiting_payment"  # awaiting_payment, paid, expired
    network: str = "Polygon Mumbai Testnet"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    paid_at: Optional[datetime] = None

class OrderCreate(BaseModel):
    eur_amount: float

class OrderResponse(BaseModel):
    id: str
    eur_amount: float
    usdc_amount: float
    payment_address: str
    status: str
    network: str
    qr_code: str
    created_at: str
    paid_at: Optional[str] = None

class PaymentCheckResponse(BaseModel):
    status: str
    paid: bool
    message: str

# Helper functions
def get_eur_to_usdc_rate():
    """Get EUR to USDC conversion rate from CoinGecko"""
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=usd-coin&vs_currencies=eur"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        # CoinGecko returns USDC price in EUR, we need EUR to USDC
        usdc_in_eur = data['usd-coin']['eur']
        # Since USDC ≈ 1 USD, and we have USDC price in EUR
        # EUR to USDC = 1 / usdc_in_eur
        return 1 / usdc_in_eur
    except Exception as e:
        logger.error(f"Error fetching EUR to USDC rate: {e}")
        # Fallback rate: approximate EUR to USD conversion
        return 1.08  # 1 EUR ≈ 1.08 USD ≈ 1.08 USDC

def generate_qr_code(address: str) -> str:
    """Generate QR code for payment address and return as base64"""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(address)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

async def check_usdc_balance(address: str, expected_amount: float) -> bool:
    """Check if wallet has received the expected USDC amount"""
    try:
        checksum_address = Web3.to_checksum_address(address)
        balance = usdc_contract.functions.balanceOf(checksum_address).call()
        # USDC has 6 decimals
        balance_usdc = balance / (10 ** 6)
        logger.info(f"Wallet {address} has {balance_usdc} USDC")
        # Allow 1% tolerance for gas/rounding
        return balance_usdc >= (expected_amount * 0.99)
    except Exception as e:
        logger.error(f"Error checking USDC balance: {e}")
        return False

# Routes
@api_router.get("/")
async def root():
    return {"message": "KryptoPay API", "network": "Polygon Mumbai Testnet"}

@api_router.post("/orders/create", response_model=OrderResponse)
async def create_order(input: OrderCreate):
    """Create a new payment order"""
    try:
        # Get conversion rate
        rate = get_eur_to_usdc_rate()
        usdc_amount = round(input.eur_amount * rate, 2)
        
        # Create order
        order = Order(
            eur_amount=input.eur_amount,
            usdc_amount=usdc_amount,
            payment_address=PAYMENT_WALLET_ADDRESS
        )
        
        # Save to database
        order_dict = order.model_dump()
        order_dict['created_at'] = order_dict['created_at'].isoformat()
        if order_dict['paid_at']:
            order_dict['paid_at'] = order_dict['paid_at'].isoformat()
        
        await db.orders.insert_one(order_dict)
        
        # Generate QR code
        qr_code = generate_qr_code(PAYMENT_WALLET_ADDRESS)
        
        return OrderResponse(
            id=order.id,
            eur_amount=order.eur_amount,
            usdc_amount=order.usdc_amount,
            payment_address=order.payment_address,
            status=order.status,
            network=order.network,
            qr_code=qr_code,
            created_at=order.created_at.isoformat(),
            paid_at=None
        )
    except Exception as e:
        logger.error(f"Error creating order: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/orders/{order_id}", response_model=OrderResponse)
async def get_order(order_id: str):
    """Get order details"""
    order = await db.orders.find_one({"id": order_id}, {"_id": 0})
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    # Generate QR code
    qr_code = generate_qr_code(order['payment_address'])
    
    return OrderResponse(
        id=order['id'],
        eur_amount=order['eur_amount'],
        usdc_amount=order['usdc_amount'],
        payment_address=order['payment_address'],
        status=order['status'],
        network=order['network'],
        qr_code=qr_code,
        created_at=order['created_at'],
        paid_at=order.get('paid_at')
    )

@api_router.post("/orders/{order_id}/check-payment", response_model=PaymentCheckResponse)
async def check_payment(order_id: str):
    """Check if payment has been received for an order"""
    order = await db.orders.find_one({"id": order_id}, {"_id": 0})
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    # If already paid, return paid status
    if order['status'] == 'paid':
        return PaymentCheckResponse(
            status='paid',
            paid=True,
            message='Payment already confirmed'
        )
    
    # Check blockchain for payment
    payment_received = await check_usdc_balance(
        order['payment_address'],
        order['usdc_amount']
    )
    
    if payment_received:
        # Update order status
        await db.orders.update_one(
            {"id": order_id},
            {
                "$set": {
                    "status": "paid",
                    "paid_at": datetime.now(timezone.utc).isoformat()
                }
            }
        )
        return PaymentCheckResponse(
            status='paid',
            paid=True,
            message='Payment confirmed!'
        )
    else:
        return PaymentCheckResponse(
            status='awaiting_payment',
            paid=False,
            message='No payment detected yet'
        )

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()