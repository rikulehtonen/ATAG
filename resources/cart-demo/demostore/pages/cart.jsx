import Image from 'next/image';
import { useSelector, useDispatch } from 'react-redux';
import { useState, useEffect } from 'react';
import { incrementQuantity, decrementQuantity, removeFromCart } from '../redux/cart.slice';
import styles from '../styles/CartPage.module.css';

const CartPage = () => {
  const cart = useSelector((state) => state.cart);
  const dispatch = useDispatch();

  const [formData, setFormData] = useState({
    name: '',
    email: '',
    address: '',
  });

  const [isFormValid, setIsFormValid] = useState(false);
  const [isPurchaseSuccessful, setIsPurchaseSuccessful] = useState(false);

  useEffect(() => {
    setIsFormValid(
      formData.name.trim() !== '' &&
        formData.email.trim() !== '' &&
        formData.address.trim() !== '' &&
        cart.length > 0
    );
  }, [formData, cart]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handlePurchase = (e) => {
    e.preventDefault();
    setIsPurchaseSuccessful(true);
    // Here, you can also add logic to handle the purchase, e.g., sending data to a server
  };

  const getTotalPrice = () => {
    return cart.reduce(
      (accumulator, item) => accumulator + item.quantity * item.price,
      0
    );
  };

  return (
    <div className={styles.container}>

      {cart.length === 0 ? (
        <h1>Your Cart is Empty!</h1>
      ) : (
        <>
          <div className={styles.header}>
            <div>Image</div>
            <div>Product</div>
            <div>Price</div>
            <div>Quantity</div>
            <div>Actions</div>
            <div>Total Price</div>
          </div>
          {cart.map((item) => (
            <div className={styles.body}>
              <div className={styles.image}>
                <Image src={item.image} height="90" width="65" />
              </div>
              <p>{item.product}</p>
              <p>$ {item.price}</p>
              <p>{item.quantity}</p>
              <div className={styles.buttons}>
                <button onClick={() => dispatch(incrementQuantity(item.id))}>+</button>
                <button onClick={() => dispatch(decrementQuantity(item.id))}>-</button>
                <button onClick={() => dispatch(removeFromCart(item.id))}>x</button>
              </div>
              <p>$ {(item.quantity * item.price).toFixed(2)}</p>
            </div>
          ))}
          <h2>{"Grand Total: $ " + getTotalPrice().toFixed(2)}</h2>

          <form onSubmit={handlePurchase} className={styles.inputs}>
            <input type="text" name="name" placeholder="Full Name" value={formData.name} onChange={handleChange} />
            <input type="email" name="email" placeholder="Email" value={formData.email} onChange={handleChange} />
            <input type="text" name="address" placeholder="Address" value={formData.address} onChange={handleChange} />
            <button type="submit" disabled={!isFormValid}>Submit Order</button>
          </form>

          {isPurchaseSuccessful && (
            <div className={styles.submitButton}>
            Purchase Successful!
            <button onClick={() => setIsPurchaseSuccessful(false)}>Close</button>
            </div>
          )}

        </>
      )}
    </div>
  );
};

export default CartPage;